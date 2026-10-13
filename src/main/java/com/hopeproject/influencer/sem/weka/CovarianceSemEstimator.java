package com.hopeproject.influencer.sem.weka;

import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

/**
 * Maximum-likelihood covariance SEM estimator for reflective latent predictors
 * and one observed numeric outcome.
 */
public final class CovarianceSemEstimator {

    private static final double LARGE_PENALTY = 1.0e100;
    private static final double MIN_VARIANCE = 1.0e-8;

    public CovarianceSemModel fit(Instances source, CompositeSpecification specification,
                                  double regularization, int maximumIterations, double tolerance) {
        validate(source, specification, regularization, maximumIterations, tolerance);
        List<Instance> rows = completeTargetRows(source);
        List<String> indicatorNames = flattenedIndicators(specification);
        int observedCount = indicatorNames.size() + 1;
        if (rows.size() < observedCount + 2) {
            throw new IllegalArgumentException("Covariance SEM requires at least " + (observedCount + 2)
                    + " rows with a known class; found " + rows.size());
        }

        double[] means = new double[observedCount];
        double[] standardDeviations = new double[observedCount];
        for (int variable = 0; variable < indicatorNames.size(); variable++) {
            Attribute attribute = source.attribute(indicatorNames.get(variable));
            means[variable] = observedMean(rows, attribute);
            standardDeviations[variable] = observedStandardDeviation(rows, attribute, means[variable]);
        }
        means[observedCount - 1] = classMean(rows);
        standardDeviations[observedCount - 1] = classStandardDeviation(rows, means[observedCount - 1]);

        double[][] standardized = standardizedData(rows, source, indicatorNames, means, standardDeviations);
        double[][] sampleCovariance = covariance(standardized);
        for (int diagonal = 0; diagonal < observedCount; diagonal++) {
            sampleCovariance[diagonal][diagonal] += regularization;
        }
        double sampleLogDeterminant;
        try {
            sampleLogDeterminant = SemMatrixOps.logDeterminantPositiveDefinite(sampleCovariance);
        } catch (IllegalArgumentException exception) {
            throw new IllegalArgumentException(
                    "The observed covariance matrix is singular; increase regularization or add independent rows",
                    exception);
        }

        Layout layout = new Layout(specification, indicatorNames);
        int covarianceMoments = observedCount * (observedCount + 1) / 2;
        int degreesOfFreedom = covarianceMoments - layout.parameterCount;
        if (degreesOfFreedom <= 0) {
            throw new IllegalArgumentException("The requested covariance model is not over-identified: "
                    + layout.parameterCount + " free parameters for " + covarianceMoments
                    + " covariance moments. Use at least three indicators per construct.");
        }

        double[] initial = initialParameters(layout, sampleCovariance);
        double fixedSampleLogDeterminant = sampleLogDeterminant;
        SemBfgsOptimizer.Result optimization = new SemBfgsOptimizer().optimize(
                parameters -> discrepancy(parameters, layout, sampleCovariance, fixedSampleLogDeterminant),
                initial, maximumIterations, tolerance);
        Decoded decoded = decode(optimization.parameters(), layout);
        double[][] implied = impliedCovariance(decoded);

        double chiSquare = Math.max(0, (rows.size() - 1.0) * optimization.value());
        double pValue = chiSquareSurvival(chiSquare, degreesOfFreedom);
        double baselineDiscrepancy = independenceDiscrepancy(sampleCovariance, sampleLogDeterminant);
        double baselineChiSquare = Math.max(0, (rows.size() - 1.0) * baselineDiscrepancy);
        int baselineDegreesOfFreedom = observedCount * (observedCount - 1) / 2;
        double cfiDenominator = Math.max(
                Math.max(baselineChiSquare - baselineDegreesOfFreedom, chiSquare - degreesOfFreedom), 1.0e-12);
        double cfi = clamp01(1.0 - Math.max(chiSquare - degreesOfFreedom, 0) / cfiDenominator);
        double tli = baselineDegreesOfFreedom == 0 || degreesOfFreedom == 0
                ? Double.NaN
                : clamp01((baselineChiSquare / baselineDegreesOfFreedom - chiSquare / degreesOfFreedom)
                / Math.max(1.0e-12, baselineChiSquare / baselineDegreesOfFreedom - 1.0));
        double rmsea = Math.sqrt(Math.max(0,
                (chiSquare - degreesOfFreedom) / (degreesOfFreedom * (rows.size() - 1.0))));
        double srmr = standardizedRootMeanSquareResidual(sampleCovariance, implied);
        double aic = chiSquare + 2.0 * layout.parameterCount;
        double bic = chiSquare + layout.parameterCount * Math.log(rows.size());
        double explainedTargetVariance = quadratic(decoded.beta, decoded.phi);
        double targetVariance = explainedTargetVariance + decoded.psi;
        double rSquared = clamp01(explainedTargetVariance / targetVariance);

        double[][] indicatorCovariance = topLeft(implied, indicatorNames.size());
        double[][] scoreCoefficients = SemMatrixOps.multiply(
                SemMatrixOps.multiply(decoded.phi, SemMatrixOps.transpose(decoded.lambda)),
                SemMatrixOps.inverse(indicatorCovariance));

        return new CovarianceSemModel(
                specification,
                indicatorNames.toArray(String[]::new),
                slice(means, 0, indicatorNames.size()),
                slice(standardDeviations, 0, indicatorNames.size()),
                means[observedCount - 1],
                standardDeviations[observedCount - 1],
                jaggedLoadings(decoded.lambda, specification),
                decoded.theta,
                decoded.phi,
                decoded.beta,
                decoded.psi,
                scoreCoefficients,
                chiSquare,
                degreesOfFreedom,
                pValue,
                cfi,
                tli,
                rmsea,
                srmr,
                aic,
                bic,
                rSquared,
                rows.size(),
                optimization.iterations(),
                optimization.converged(),
                optimization.gradientNorm(),
                regularization);
    }

    public CovarianceSemModel fit(Instances source, CompositeSpecification specification) {
        return fit(source, specification, 1.0e-6, 300, 1.0e-6);
    }

    private void validate(Instances source, CompositeSpecification specification,
                          double regularization, int maximumIterations, double tolerance) {
        if (source == null) throw new IllegalArgumentException("Training data cannot be null");
        if (source.classIndex() < 0 || !source.classAttribute().isNumeric()) {
            throw new IllegalArgumentException("A numeric class attribute is required");
        }
        if (!Double.isFinite(regularization) || regularization <= 0) {
            throw new IllegalArgumentException("Regularization must be finite and strictly positive");
        }
        if (maximumIterations < 1) throw new IllegalArgumentException("Maximum iterations must be positive");
        if (!Double.isFinite(tolerance) || tolerance <= 0) {
            throw new IllegalArgumentException("Tolerance must be finite and strictly positive");
        }
        Set<String> indicators = new HashSet<>();
        for (CompositeSpecification.Construct construct : specification.constructs()) {
            if (construct.indicators().size() < 3) {
                throw new IllegalArgumentException("Covariance SEM requires at least three reflective indicators for "
                        + construct.name());
            }
            for (String name : construct.indicators()) {
                Attribute attribute = source.attribute(name);
                if (attribute == null || !attribute.isNumeric() || attribute.index() == source.classIndex()) {
                    throw new IllegalArgumentException("Unknown or non-numeric indicator attribute: " + name);
                }
                if (!indicators.add(name.toLowerCase())) {
                    throw new IllegalArgumentException("Indicators cannot load on several constructs: " + name);
                }
            }
        }
    }

    private List<Instance> completeTargetRows(Instances source) {
        List<Instance> rows = new ArrayList<>();
        for (Instance instance : source) if (!instance.classIsMissing()) rows.add(instance);
        return rows;
    }

    private List<String> flattenedIndicators(CompositeSpecification specification) {
        List<String> result = new ArrayList<>();
        for (CompositeSpecification.Construct construct : specification.constructs()) {
            result.addAll(construct.indicators());
        }
        return result;
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

    private double observedStandardDeviation(List<Instance> rows, Attribute attribute, double mean) {
        double total = 0;
        int count = 0;
        for (Instance row : rows) {
            if (!row.isMissing(attribute)) {
                double difference = row.value(attribute) - mean;
                total += difference * difference;
                count++;
            }
        }
        double result = Math.sqrt(total / Math.max(1, count - 1.0));
        if (!(result > MIN_VARIANCE)) {
            throw new IllegalArgumentException("Indicator has no variance: " + attribute.name());
        }
        return result;
    }

    private double classMean(List<Instance> rows) {
        double total = 0;
        for (Instance row : rows) total += row.classValue();
        return total / rows.size();
    }

    private double classStandardDeviation(List<Instance> rows, double mean) {
        double total = 0;
        for (Instance row : rows) {
            double difference = row.classValue() - mean;
            total += difference * difference;
        }
        double result = Math.sqrt(total / Math.max(1, rows.size() - 1.0));
        if (!(result > MIN_VARIANCE)) throw new IllegalArgumentException("The class attribute has no variance");
        return result;
    }

    private double[][] standardizedData(List<Instance> rows, Instances source, List<String> indicators,
                                        double[] means, double[] standardDeviations) {
        double[][] result = new double[rows.size()][indicators.size() + 1];
        for (int row = 0; row < rows.size(); row++) {
            Instance instance = rows.get(row);
            for (int variable = 0; variable < indicators.size(); variable++) {
                Attribute attribute = source.attribute(indicators.get(variable));
                result[row][variable] = instance.isMissing(attribute)
                        ? 0.0
                        : (instance.value(attribute) - means[variable]) / standardDeviations[variable];
            }
            result[row][indicators.size()] =
                    (instance.classValue() - means[indicators.size()]) / standardDeviations[indicators.size()];
        }
        return result;
    }

    private double[][] covariance(double[][] data) {
        double[][] result = new double[data[0].length][data[0].length];
        for (double[] row : data) {
            for (int left = 0; left < row.length; left++) {
                for (int right = 0; right < row.length; right++) result[left][right] += row[left] * row[right];
            }
        }
        for (int left = 0; left < result.length; left++) {
            for (int right = 0; right < result.length; right++) result[left][right] /= data.length - 1.0;
        }
        return result;
    }

    private double[] initialParameters(Layout layout, double[][] sampleCovariance) {
        double[] result = new double[layout.parameterCount];
        int cursor = 0;
        for (int construct = 0; construct < layout.constructCount; construct++) {
            int marker = layout.firstIndicator[construct];
            int end = marker + layout.indicatorCounts[construct];
            for (int indicator = marker + 1; indicator < end; indicator++) {
                result[cursor++] = clipped(sampleCovariance[indicator][marker], -0.9, 0.9);
            }
        }
        for (int indicator = 0; indicator < layout.indicatorCount; indicator++) result[cursor++] = Math.log(0.5);
        for (int row = 0; row < layout.constructCount; row++) {
            for (int column = 0; column <= row; column++) {
                result[cursor++] = row == column ? Math.log(Math.sqrt(0.5)) : 0.0;
            }
        }
        int target = layout.indicatorCount;
        for (int construct = 0; construct < layout.constructCount; construct++) {
            result[cursor++] = clipped(sampleCovariance[layout.firstIndicator[construct]][target], -0.8, 0.8);
        }
        result[cursor] = Math.log(0.5);
        return result;
    }

    private double discrepancy(double[] parameters, Layout layout, double[][] sampleCovariance,
                               double sampleLogDeterminant) {
        try {
            double[][] implied = impliedCovariance(decode(parameters, layout));
            double value = SemMatrixOps.logDeterminantPositiveDefinite(implied)
                    + SemMatrixOps.traceProduct(sampleCovariance, SemMatrixOps.inverse(implied))
                    - sampleLogDeterminant - implied.length;
            return Double.isFinite(value) ? Math.max(0, value) : LARGE_PENALTY;
        } catch (RuntimeException exception) {
            return LARGE_PENALTY;
        }
    }

    private Decoded decode(double[] parameters, Layout layout) {
        int cursor = 0;
        double[][] lambda = new double[layout.indicatorCount][layout.constructCount];
        for (int construct = 0; construct < layout.constructCount; construct++) {
            int marker = layout.firstIndicator[construct];
            lambda[marker][construct] = 1.0;
            int end = marker + layout.indicatorCounts[construct];
            for (int indicator = marker + 1; indicator < end; indicator++) {
                lambda[indicator][construct] = clipped(parameters[cursor++], -5.0, 5.0);
            }
        }
        double[] theta = new double[layout.indicatorCount];
        for (int indicator = 0; indicator < theta.length; indicator++) theta[indicator] = positive(parameters[cursor++]);
        double[][] cholesky = new double[layout.constructCount][layout.constructCount];
        for (int row = 0; row < layout.constructCount; row++) {
            for (int column = 0; column <= row; column++) {
                cholesky[row][column] = row == column
                        ? Math.sqrt(positive(2.0 * parameters[cursor++]))
                        : clipped(parameters[cursor++], -5.0, 5.0);
            }
        }
        double[][] phi = SemMatrixOps.multiply(cholesky, SemMatrixOps.transpose(cholesky));
        double[] beta = new double[layout.constructCount];
        for (int construct = 0; construct < beta.length; construct++) {
            beta[construct] = clipped(parameters[cursor++], -10.0, 10.0);
        }
        double psi = positive(parameters[cursor]);
        return new Decoded(lambda, theta, phi, beta, psi);
    }

    private double positive(double logValue) {
        return Math.exp(clipped(logValue, -18.0, 18.0)) + MIN_VARIANCE;
    }

    private double[][] impliedCovariance(Decoded decoded) {
        int indicators = decoded.lambda.length;
        double[][] indicatorCovariance = SemMatrixOps.multiply(
                SemMatrixOps.multiply(decoded.lambda, decoded.phi), SemMatrixOps.transpose(decoded.lambda));
        for (int indicator = 0; indicator < indicators; indicator++) {
            indicatorCovariance[indicator][indicator] += decoded.theta[indicator];
        }
        double[] latentTargetCovariance = SemMatrixOps.multiply(decoded.phi, decoded.beta);
        double[] indicatorTargetCovariance = SemMatrixOps.multiply(decoded.lambda, latentTargetCovariance);
        double targetVariance = quadratic(decoded.beta, decoded.phi) + decoded.psi;
        double[][] result = new double[indicators + 1][indicators + 1];
        for (int row = 0; row < indicators; row++) {
            System.arraycopy(indicatorCovariance[row], 0, result[row], 0, indicators);
            result[row][indicators] = indicatorTargetCovariance[row];
            result[indicators][row] = indicatorTargetCovariance[row];
        }
        result[indicators][indicators] = targetVariance;
        return result;
    }

    private double independenceDiscrepancy(double[][] sample, double sampleLogDeterminant) {
        double logDeterminant = 0;
        double trace = 0;
        for (int index = 0; index < sample.length; index++) {
            logDeterminant += Math.log(sample[index][index]);
            trace += sample[index][index] / sample[index][index];
        }
        return Math.max(0, logDeterminant + trace - sampleLogDeterminant - sample.length);
    }

    private double standardizedRootMeanSquareResidual(double[][] observed, double[][] implied) {
        double total = 0;
        int count = 0;
        for (int row = 0; row < observed.length; row++) {
            for (int column = 0; column <= row; column++) {
                double observedCorrelation = observed[row][column]
                        / Math.sqrt(observed[row][row] * observed[column][column]);
                double impliedCorrelation = implied[row][column]
                        / Math.sqrt(implied[row][row] * implied[column][column]);
                double difference = observedCorrelation - impliedCorrelation;
                total += difference * difference;
                count++;
            }
        }
        return Math.sqrt(total / count);
    }

    private double quadratic(double[] vector, double[][] matrix) {
        return SemMatrixOps.dot(vector, SemMatrixOps.multiply(matrix, vector));
    }

    private double[][] topLeft(double[][] source, int size) {
        double[][] result = new double[size][size];
        for (int row = 0; row < size; row++) System.arraycopy(source[row], 0, result[row], 0, size);
        return result;
    }

    private double[][] jaggedLoadings(double[][] lambda, CompositeSpecification specification) {
        double[][] result = new double[specification.constructs().size()][];
        int indicator = 0;
        for (int construct = 0; construct < result.length; construct++) {
            result[construct] = new double[specification.constructs().get(construct).indicators().size()];
            for (int local = 0; local < result[construct].length; local++) {
                result[construct][local] = lambda[indicator++][construct];
            }
        }
        return result;
    }

    private double[] slice(double[] source, int from, int to) {
        double[] result = new double[to - from];
        System.arraycopy(source, from, result, 0, result.length);
        return result;
    }

    private double clipped(double value, double minimum, double maximum) {
        return Math.max(minimum, Math.min(maximum, value));
    }

    private double clamp01(double value) {
        return Math.max(0, Math.min(1, value));
    }

    private double chiSquareSurvival(double value, int degreesOfFreedom) {
        if (value <= 0) return 1.0;
        return regularizedGammaQ(degreesOfFreedom / 2.0, value / 2.0);
    }

    private double regularizedGammaQ(double a, double x) {
        if (x < a + 1.0) return 1.0 - regularizedGammaPSeries(a, x);
        double b = x + 1.0 - a;
        double c = 1.0 / 1.0e-300;
        double d = 1.0 / b;
        double h = d;
        for (int iteration = 1; iteration <= 200; iteration++) {
            double an = -iteration * (iteration - a);
            b += 2.0;
            d = an * d + b;
            if (Math.abs(d) < 1.0e-300) d = 1.0e-300;
            c = b + an / c;
            if (Math.abs(c) < 1.0e-300) c = 1.0e-300;
            d = 1.0 / d;
            double delta = d * c;
            h *= delta;
            if (Math.abs(delta - 1.0) < 1.0e-12) break;
        }
        return Math.exp(-x + a * Math.log(x) - logGamma(a)) * h;
    }

    private double regularizedGammaPSeries(double a, double x) {
        if (x <= 0) return 0;
        double sum = 1.0 / a;
        double term = sum;
        double ap = a;
        for (int iteration = 1; iteration <= 200; iteration++) {
            ap += 1.0;
            term *= x / ap;
            sum += term;
            if (Math.abs(term) < Math.abs(sum) * 1.0e-12) break;
        }
        return sum * Math.exp(-x + a * Math.log(x) - logGamma(a));
    }

    private double logGamma(double value) {
        double[] coefficients = {
                676.5203681218851, -1259.1392167224028, 771.32342877765313,
                -176.61502916214059, 12.507343278686905, -0.13857109526572012,
                9.9843695780195716e-6, 1.5056327351493116e-7
        };
        if (value < 0.5) return Math.log(Math.PI) - Math.log(Math.sin(Math.PI * value)) - logGamma(1 - value);
        double shifted = value - 1;
        double series = 0.99999999999980993;
        for (int index = 0; index < coefficients.length; index++) series += coefficients[index] / (shifted + index + 1);
        double t = shifted + coefficients.length - 0.5;
        return 0.5 * Math.log(2 * Math.PI) + (shifted + 0.5) * Math.log(t) - t + Math.log(series);
    }

    private record Decoded(double[][] lambda, double[] theta, double[][] phi, double[] beta, double psi) {
    }

    private static final class Layout {
        private final int constructCount;
        private final int indicatorCount;
        private final int[] firstIndicator;
        private final int[] indicatorCounts;
        private final int parameterCount;

        private Layout(CompositeSpecification specification, List<String> indicators) {
            constructCount = specification.constructs().size();
            indicatorCount = indicators.size();
            firstIndicator = new int[constructCount];
            indicatorCounts = new int[constructCount];
            int cursor = 0;
            for (int construct = 0; construct < constructCount; construct++) {
                firstIndicator[construct] = cursor;
                indicatorCounts[construct] = specification.constructs().get(construct).indicators().size();
                cursor += indicatorCounts[construct];
            }
            int freeLoadings = indicatorCount - constructCount;
            int latentCholesky = constructCount * (constructCount + 1) / 2;
            parameterCount = freeLoadings + indicatorCount + latentCholesky + constructCount + 1;
        }
    }
}
