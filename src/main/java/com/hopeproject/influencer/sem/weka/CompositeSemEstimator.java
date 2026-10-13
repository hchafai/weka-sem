package com.hopeproject.influencer.sem.weka;

import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;

import java.util.ArrayList;
import java.util.List;

/** Fits a configurable equal- or fixed-weight composite path model to Weka instances. */
public final class CompositeSemEstimator {

    private static final double EPSILON = 1.0e-12;

    public CompositeSemModel fit(Instances source, CompositeSpecification specification, double ridge) {
        if (source == null) throw new IllegalArgumentException("Training data cannot be null");
        if (source.classIndex() < 0) throw new IllegalArgumentException("A numeric class attribute is required");
        if (!source.classAttribute().isNumeric()) {
            throw new IllegalArgumentException("Composite path regression requires a numeric class attribute");
        }
        if (!Double.isFinite(ridge) || ridge < 0) {
            throw new IllegalArgumentException("Ridge must be a finite non-negative value");
        }
        validateIndicators(source, specification);

        List<Instance> rows = new ArrayList<>();
        for (Instance instance : source) {
            if (!instance.classIsMissing()) rows.add(instance);
        }
        int constructCount = specification.constructs().size();
        if (rows.size() <= constructCount + 1) {
            throw new IllegalArgumentException(
                    "At least " + (constructCount + 2) + " rows with a known class are required; found " + rows.size());
        }

        double[][] indicatorMeans = new double[constructCount][];
        double[][] indicatorStd = new double[constructCount][];
        double[][][] standardizedIndicators = new double[constructCount][][];
        double[][] rawConstructs = new double[rows.size()][constructCount];

        for (int constructIndex = 0; constructIndex < constructCount; constructIndex++) {
            CompositeSpecification.Construct construct = specification.constructs().get(constructIndex);
            int indicatorCount = construct.indicators().size();
            indicatorMeans[constructIndex] = new double[indicatorCount];
            indicatorStd[constructIndex] = new double[indicatorCount];
            standardizedIndicators[constructIndex] = new double[rows.size()][indicatorCount];
            double[] weights = construct.weights();

            for (int indicatorIndex = 0; indicatorIndex < indicatorCount; indicatorIndex++) {
                Attribute attribute = source.attribute(construct.indicators().get(indicatorIndex));
                indicatorMeans[constructIndex][indicatorIndex] = observedMean(rows, attribute);
                indicatorStd[constructIndex][indicatorIndex] = observedStd(
                        rows, attribute, indicatorMeans[constructIndex][indicatorIndex]);
                for (int row = 0; row < rows.size(); row++) {
                    Instance instance = rows.get(row);
                    double standardized = instance.isMissing(attribute)
                            ? 0.0
                            : (instance.value(attribute) - indicatorMeans[constructIndex][indicatorIndex])
                            / safeStd(indicatorStd[constructIndex][indicatorIndex]);
                    standardizedIndicators[constructIndex][row][indicatorIndex] = standardized;
                    rawConstructs[row][constructIndex] += weights[indicatorIndex] * standardized;
                }
            }
        }

        double[] constructMeans = columnMeans(rawConstructs);
        double[] constructStd = columnStandardDeviations(rawConstructs, constructMeans);
        double[][] constructs = standardize(rawConstructs, constructMeans, constructStd);

        double[] target = new double[rows.size()];
        for (int row = 0; row < rows.size(); row++) target[row] = rows.get(row).classValue();
        double targetMean = mean(target);
        double targetStd = standardDeviation(target, targetMean);
        if (targetStd < EPSILON) throw new IllegalArgumentException("The class attribute has no variance");
        double[] standardizedTarget = new double[target.length];
        for (int row = 0; row < target.length; row++) {
            standardizedTarget[row] = (target[row] - targetMean) / targetStd;
        }

        double[][] xtx = crossProduct(constructs);
        double[] xty = crossProduct(constructs, standardizedTarget);
        for (int i = 0; i < constructCount; i++) xtx[i][i] += ridge;
        double[][] inverse = inverse(xtx);
        double[] paths = multiply(inverse, xty);
        double[] predicted = multiply(constructs, paths);

        double residualSumSquares = 0;
        double totalSumSquares = 0;
        for (int row = 0; row < target.length; row++) {
            double residual = standardizedTarget[row] - predicted[row];
            residualSumSquares += residual * residual;
            totalSumSquares += standardizedTarget[row] * standardizedTarget[row];
        }
        double rSquared = totalSumSquares < EPSILON ? 0 : 1 - residualSumSquares / totalSumSquares;
        double adjusted = rows.size() <= constructCount + 1 ? rSquared
                : 1 - (1 - rSquared) * (rows.size() - 1.0) / (rows.size() - constructCount - 1.0);
        double rmse = Math.sqrt(residualSumSquares / rows.size());

        double residualVariance = residualSumSquares / Math.max(1, rows.size() - constructCount);
        double[] standardErrors = new double[constructCount];
        double[] pValues = new double[constructCount];
        for (int i = 0; i < constructCount; i++) {
            standardErrors[i] = Math.sqrt(Math.max(0, residualVariance * inverse[i][i]));
            double z = standardErrors[i] < EPSILON ? 0 : paths[i] / standardErrors[i];
            pValues[i] = Math.max(0, Math.min(1, 2 * (1 - normalCdf(Math.abs(z)))));
        }

        double[][] loadings = measurementLoadings(standardizedIndicators, constructs);
        double[] reliability = compositeReliability(loadings);
        double srmr = srmr(standardizedIndicators, standardizedTarget, constructs, loadings, paths);

        return new CompositeSemModel(
                specification, indicatorMeans, indicatorStd, constructMeans, constructStd,
                loadings, reliability, paths, standardErrors, pValues,
                targetMean, targetStd, rSquared, adjusted, rmse, srmr, rows.size(), ridge);
    }

    private void validateIndicators(Instances data, CompositeSpecification specification) {
        for (CompositeSpecification.Construct construct : specification.constructs()) {
            for (String indicatorName : construct.indicators()) {
                Attribute attribute = data.attribute(indicatorName);
                if (attribute == null) {
                    throw new IllegalArgumentException("Unknown indicator attribute: " + indicatorName);
                }
                if (attribute.index() == data.classIndex()) {
                    throw new IllegalArgumentException("The class cannot also be an indicator: " + indicatorName);
                }
                if (!attribute.isNumeric()) {
                    throw new IllegalArgumentException("Indicator attribute must be numeric: " + indicatorName);
                }
            }
        }
    }

    private double observedMean(List<Instance> rows, Attribute attribute) {
        double total = 0;
        int count = 0;
        for (Instance row : rows) {
            if (!row.isMissing(attribute)) {
                total += row.value(attribute);
                count++;
            }
        }
        if (count == 0) throw new IllegalArgumentException("Indicator contains only missing values: " + attribute.name());
        return total / count;
    }

    private double observedStd(List<Instance> rows, Attribute attribute, double mean) {
        double total = 0;
        int count = 0;
        for (Instance row : rows) {
            if (!row.isMissing(attribute)) {
                double delta = row.value(attribute) - mean;
                total += delta * delta;
                count++;
            }
        }
        return count <= 1 ? 1.0 : safeStd(Math.sqrt(total / (count - 1.0)));
    }

    private double[][] measurementLoadings(double[][][] indicators, double[][] constructs) {
        double[][] loadings = new double[indicators.length][];
        for (int construct = 0; construct < indicators.length; construct++) {
            loadings[construct] = new double[indicators[construct][0].length];
            double[] constructColumn = column(constructs, construct);
            for (int indicator = 0; indicator < loadings[construct].length; indicator++) {
                loadings[construct][indicator] = correlation(column(indicators[construct], indicator), constructColumn);
            }
        }
        return loadings;
    }

    private double[] compositeReliability(double[][] loadings) {
        double[] result = new double[loadings.length];
        for (int construct = 0; construct < loadings.length; construct++) {
            double sum = 0;
            double error = 0;
            for (double loading : loadings[construct]) {
                sum += loading;
                error += Math.max(0, 1 - loading * loading);
            }
            result[construct] = (sum * sum) / Math.max(EPSILON, sum * sum + error);
        }
        return result;
    }

    private double srmr(double[][][] indicators, double[] target, double[][] constructs,
                        double[][] loadings, double[] paths) {
        int totalIndicators = 0;
        for (double[][] constructIndicators : indicators) totalIndicators += constructIndicators[0].length;
        double[][] flattened = new double[totalIndicators][];
        int[] owner = new int[totalIndicators];
        int[] localIndex = new int[totalIndicators];
        int cursor = 0;
        for (int construct = 0; construct < indicators.length; construct++) {
            for (int indicator = 0; indicator < indicators[construct][0].length; indicator++) {
                flattened[cursor] = column(indicators[construct], indicator);
                owner[cursor] = construct;
                localIndex[cursor] = indicator;
                cursor++;
            }
        }

        double[][] constructCorrelation = new double[constructs[0].length][constructs[0].length];
        for (int left = 0; left < constructCorrelation.length; left++) {
            for (int right = 0; right < constructCorrelation.length; right++) {
                constructCorrelation[left][right] = correlation(column(constructs, left), column(constructs, right));
            }
        }
        double[] constructTarget = multiply(constructCorrelation, paths);

        double squaredResiduals = 0;
        int pairs = 0;
        for (int left = 0; left < flattened.length; left++) {
            for (int right = left + 1; right < flattened.length; right++) {
                double observed = correlation(flattened[left], flattened[right]);
                double implied = loadings[owner[left]][localIndex[left]]
                        * loadings[owner[right]][localIndex[right]]
                        * constructCorrelation[owner[left]][owner[right]];
                double residual = observed - implied;
                squaredResiduals += residual * residual;
                pairs++;
            }
            double observedTarget = correlation(flattened[left], target);
            double impliedTarget = loadings[owner[left]][localIndex[left]] * constructTarget[owner[left]];
            double residualTarget = observedTarget - impliedTarget;
            squaredResiduals += residualTarget * residualTarget;
            pairs++;
        }
        return Math.sqrt(squaredResiduals / Math.max(1, pairs));
    }

    private double[] columnMeans(double[][] values) {
        double[] means = new double[values[0].length];
        for (double[] row : values) {
            for (int column = 0; column < row.length; column++) means[column] += row[column];
        }
        for (int column = 0; column < means.length; column++) means[column] /= values.length;
        return means;
    }

    private double[] columnStandardDeviations(double[][] values, double[] means) {
        double[] std = new double[means.length];
        for (double[] row : values) {
            for (int column = 0; column < row.length; column++) {
                double delta = row[column] - means[column];
                std[column] += delta * delta;
            }
        }
        for (int column = 0; column < std.length; column++) {
            std[column] = safeStd(Math.sqrt(std[column] / Math.max(1, values.length - 1)));
        }
        return std;
    }

    private double[][] standardize(double[][] values, double[] means, double[] std) {
        double[][] result = new double[values.length][values[0].length];
        for (int row = 0; row < values.length; row++) {
            for (int column = 0; column < values[row].length; column++) {
                result[row][column] = (values[row][column] - means[column]) / safeStd(std[column]);
            }
        }
        return result;
    }

    private double[][] crossProduct(double[][] x) {
        double[][] result = new double[x[0].length][x[0].length];
        for (double[] row : x) {
            for (int i = 0; i < row.length; i++) {
                for (int j = 0; j < row.length; j++) result[i][j] += row[i] * row[j];
            }
        }
        return result;
    }

    private double[] crossProduct(double[][] x, double[] y) {
        double[] result = new double[x[0].length];
        for (int row = 0; row < x.length; row++) {
            for (int column = 0; column < x[row].length; column++) result[column] += x[row][column] * y[row];
        }
        return result;
    }

    private double[][] inverse(double[][] matrix) {
        int size = matrix.length;
        double[][] augmented = new double[size][size * 2];
        for (int i = 0; i < size; i++) {
            System.arraycopy(matrix[i], 0, augmented[i], 0, size);
            augmented[i][size + i] = 1;
        }
        for (int pivot = 0; pivot < size; pivot++) {
            int best = pivot;
            for (int row = pivot + 1; row < size; row++) {
                if (Math.abs(augmented[row][pivot]) > Math.abs(augmented[best][pivot])) best = row;
            }
            if (Math.abs(augmented[best][pivot]) < EPSILON) {
                throw new IllegalArgumentException("The composite score matrix is singular; increase ridge or revise the constructs");
            }
            double[] swap = augmented[pivot];
            augmented[pivot] = augmented[best];
            augmented[best] = swap;
            double divisor = augmented[pivot][pivot];
            for (int column = 0; column < size * 2; column++) augmented[pivot][column] /= divisor;
            for (int row = 0; row < size; row++) {
                if (row == pivot) continue;
                double factor = augmented[row][pivot];
                for (int column = 0; column < size * 2; column++) {
                    augmented[row][column] -= factor * augmented[pivot][column];
                }
            }
        }
        double[][] inverse = new double[size][size];
        for (int i = 0; i < size; i++) System.arraycopy(augmented[i], size, inverse[i], 0, size);
        return inverse;
    }

    private double[] multiply(double[][] matrix, double[] vector) {
        double[] result = new double[matrix.length];
        for (int row = 0; row < matrix.length; row++) {
            for (int column = 0; column < vector.length; column++) {
                result[row] += matrix[row][column] * vector[column];
            }
        }
        return result;
    }

    private double[] column(double[][] values, int column) {
        double[] result = new double[values.length];
        for (int row = 0; row < values.length; row++) result[row] = values[row][column];
        return result;
    }

    private double correlation(double[] left, double[] right) {
        double leftMean = mean(left);
        double rightMean = mean(right);
        double covariance = 0;
        double leftVariance = 0;
        double rightVariance = 0;
        for (int i = 0; i < left.length; i++) {
            double leftDelta = left[i] - leftMean;
            double rightDelta = right[i] - rightMean;
            covariance += leftDelta * rightDelta;
            leftVariance += leftDelta * leftDelta;
            rightVariance += rightDelta * rightDelta;
        }
        double denominator = Math.sqrt(leftVariance * rightVariance);
        return denominator < EPSILON ? 0.0 : covariance / denominator;
    }

    private double mean(double[] values) {
        double total = 0;
        for (double value : values) total += value;
        return total / values.length;
    }

    private double standardDeviation(double[] values, double mean) {
        double total = 0;
        for (double value : values) {
            double delta = value - mean;
            total += delta * delta;
        }
        return Math.sqrt(total / Math.max(1, values.length - 1));
    }

    private double safeStd(double value) {
        return Math.abs(value) < EPSILON ? 1.0 : value;
    }

    private double normalCdf(double value) {
        return 0.5 * (1 + erf(value / Math.sqrt(2)));
    }

    private double erf(double value) {
        double sign = value < 0 ? -1 : 1;
        double x = Math.abs(value);
        double t = 1 / (1 + 0.3275911 * x);
        double polynomial = (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t
                - 0.284496736) * t + 0.254829592) * t;
        return sign * (1 - polynomial * Math.exp(-x * x));
    }
}
