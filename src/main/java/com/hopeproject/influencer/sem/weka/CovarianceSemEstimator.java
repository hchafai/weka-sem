package com.hopeproject.influencer.sem.weka;

import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.function.ToDoubleFunction;

/**
 * Maximum-likelihood covariance SEM estimator for reflective latent predictors
 * and one observed numeric outcome.
 */
public final class CovarianceSemEstimator {

    private static final double LARGE_PENALTY = 1.0e100;
    private static final double MIN_VARIANCE = 1.0e-8;

    /** Fits a model and computes observed-information parameter inference. */
    public CovarianceSemModel fit(Instances source, CompositeSpecification specification,
                                  double regularization, int maximumIterations, double tolerance) {
        return fit(source, specification, regularization, maximumIterations, tolerance, true);
    }

    /**
     * Fits the covariance model and optionally computes the observed-information
     * matrix used for Wald inference.
     *
     * @param source Weka data with a numeric class
     * @param specification reflective measurement specification
     * @param regularization positive ridge added only during optimization
     * @param maximumIterations maximum BFGS iterations
     * @param tolerance numerical convergence tolerance
     * @param computeInference whether to compute standard errors, z statistics and p-values
     * @return immutable fitted model
     */
    public CovarianceSemModel fit(Instances source, CompositeSpecification specification,
                                  double regularization, int maximumIterations, double tolerance,
                                  boolean computeInference) {
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
        double[][] fittingCovariance = SemMatrixOps.copy(sampleCovariance);
        for (int diagonal = 0; diagonal < observedCount; diagonal++) {
            fittingCovariance[diagonal][diagonal] += regularization;
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

        double fittingLogDeterminant = SemMatrixOps.logDeterminantPositiveDefinite(fittingCovariance);
        double[] initial = initialParameters(layout, fittingCovariance);
        SemBfgsOptimizer.Result optimization = new SemBfgsOptimizer().optimize(
                parameters -> discrepancy(parameters, layout, fittingCovariance, fittingLogDeterminant),
                initial, maximumIterations, tolerance);
        Decoded decoded = decode(optimization.parameters(), layout);
        double[][] implied = impliedCovariance(decoded);

        double fittedDiscrepancy = discrepancy(
                optimization.parameters(), layout, sampleCovariance, sampleLogDeterminant);
        double chiSquare = Math.max(0, rows.size() * fittedDiscrepancy);
        double pValue = chiSquareSurvival(chiSquare, degreesOfFreedom);
        double baselineDiscrepancy = independenceDiscrepancy(sampleCovariance, sampleLogDeterminant);
        double baselineChiSquare = Math.max(0, rows.size() * baselineDiscrepancy);
        int baselineDegreesOfFreedom = observedCount * (observedCount - 1) / 2;
        double cfiDenominator = Math.max(
                Math.max(baselineChiSquare - baselineDegreesOfFreedom, chiSquare - degreesOfFreedom), 1.0e-12);
        double cfi = clamp01(1.0 - Math.max(chiSquare - degreesOfFreedom, 0) / cfiDenominator);
        double tli = baselineDegreesOfFreedom == 0 || degreesOfFreedom == 0
                ? Double.NaN
                : clamp01((baselineChiSquare / baselineDegreesOfFreedom - chiSquare / degreesOfFreedom)
                / Math.max(1.0e-12, baselineChiSquare / baselineDegreesOfFreedom - 1.0));
        double rmsea = Math.sqrt(Math.max(0,
                (chiSquare - degreesOfFreedom) / (degreesOfFreedom * rows.size())));
        double srmr = standardizedRootMeanSquareResidual(sampleCovariance, implied);
        double negativeTwoLogLikelihood = negativeTwoLogLikelihood(
                implied, sampleCovariance, standardDeviations, rows.size());
        double aic = negativeTwoLogLikelihood + 2.0 * layout.parameterCount;
        double bic = negativeTwoLogLikelihood + layout.parameterCount * Math.log(rows.size());
        double explainedTargetVariance = quadratic(decoded.beta, decoded.phi);
        double targetVariance = explainedTargetVariance + decoded.psi;
        double rSquared = clamp01(explainedTargetVariance / targetVariance);

        double[][] standardizedLoadings = standardizedLoadings(decoded, specification);
        double[] standardizedResidualVariances = standardizedResidualVariances(decoded);
        double[][] latentCorrelations = latentCorrelations(decoded.phi);
        double[] standardizedPaths = standardizedPaths(decoded, targetVariance);
        double standardizedTargetResidualVariance = decoded.psi / targetVariance;
        List<SemParameterEstimate> parameterEstimates = computeInference
                ? parameterEstimates(optimization.parameters(), layout, specification,
                standardDeviations, rows.size(), source.classAttribute().name(),
                parameters -> discrepancy(parameters, layout, fittingCovariance, fittingLogDeterminant),
                optimization.inverseHessian())
                : List.of();

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
                standardizedLoadings,
                decoded.theta,
                standardizedResidualVariances,
                decoded.phi,
                latentCorrelations,
                decoded.beta,
                standardizedPaths,
                decoded.psi,
                standardizedTargetResidualVariance,
                scoreCoefficients,
                parameterEstimates,
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

    /** Fits a model with the documented default numerical settings. */
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

    private double negativeTwoLogLikelihood(double[][] implied, double[][] observedCorrelation,
                                            double[] sampleStandardDeviations, int sampleSize) {
        double mlScale = Math.sqrt((sampleSize - 1.0) / sampleSize);
        double[] mlStandardDeviations = new double[sampleStandardDeviations.length];
        for (int index = 0; index < mlStandardDeviations.length; index++) {
            mlStandardDeviations[index] = sampleStandardDeviations[index] * mlScale;
        }
        double[][] rawImplied = rescaleCovariance(implied, mlStandardDeviations);
        double[][] rawObserved = rescaleCovariance(observedCorrelation, mlStandardDeviations);
        return sampleSize * (implied.length * Math.log(2.0 * Math.PI)
                + SemMatrixOps.logDeterminantPositiveDefinite(rawImplied)
                + SemMatrixOps.traceProduct(rawObserved, SemMatrixOps.inverse(rawImplied)));
    }

    private double[][] rescaleCovariance(double[][] covariance, double[] scales) {
        double[][] result = new double[covariance.length][covariance.length];
        for (int row = 0; row < covariance.length; row++) {
            for (int column = 0; column < covariance.length; column++) {
                result[row][column] = covariance[row][column] * scales[row] * scales[column];
            }
        }
        return result;
    }

    private double[][] standardizedLoadings(Decoded decoded, CompositeSpecification specification) {
        double[][] implied = impliedCovariance(decoded);
        double[][] result = new double[specification.constructs().size()][];
        int indicator = 0;
        for (int construct = 0; construct < result.length; construct++) {
            result[construct] = new double[specification.constructs().get(construct).indicators().size()];
            for (int local = 0; local < result[construct].length; local++) {
                result[construct][local] = decoded.lambda[indicator][construct]
                        * Math.sqrt(decoded.phi[construct][construct] / implied[indicator][indicator]);
                indicator++;
            }
        }
        return result;
    }

    private double[] standardizedResidualVariances(Decoded decoded) {
        double[][] implied = impliedCovariance(decoded);
        double[] result = new double[decoded.theta.length];
        for (int indicator = 0; indicator < result.length; indicator++) {
            result[indicator] = decoded.theta[indicator] / implied[indicator][indicator];
        }
        return result;
    }

    private double[][] latentCorrelations(double[][] covariance) {
        double[][] result = new double[covariance.length][covariance.length];
        for (int row = 0; row < covariance.length; row++) {
            for (int column = 0; column < covariance.length; column++) {
                result[row][column] = covariance[row][column]
                        / Math.sqrt(covariance[row][row] * covariance[column][column]);
            }
        }
        return result;
    }

    private double[] standardizedPaths(Decoded decoded, double targetVariance) {
        double[] result = new double[decoded.beta.length];
        for (int construct = 0; construct < result.length; construct++) {
            result[construct] = decoded.beta[construct]
                    * Math.sqrt(decoded.phi[construct][construct] / targetVariance);
        }
        return result;
    }

    private List<SemParameterEstimate> parameterEstimates(
            double[] optimum,
            Layout layout,
            CompositeSpecification specification,
            double[] sampleStandardDeviations,
            int sampleSize,
            String targetName,
            ToDoubleFunction<double[]> objective,
            double[][] fallbackInverseHessian) {
        List<ParameterDescriptor> descriptors = parameterDescriptors(specification, targetName);
        double[] estimates = reportedParameterValues(
                optimum, layout, specification, sampleStandardDeviations, sampleSize, false);
        double[] standardized = reportedParameterValues(
                optimum, layout, specification, sampleStandardDeviations, sampleSize, true);
        double[][] inverseHessian = inverseObservedHessian(objective, optimum, fallbackInverseHessian);
        double covarianceScale = 2.0 / sampleSize;

        List<SemParameterEstimate> result = new ArrayList<>(descriptors.size());
        for (int parameter = 0; parameter < descriptors.size(); parameter++) {
            ParameterDescriptor descriptor = descriptors.get(parameter);
            if (!descriptor.free) {
                result.add(new SemParameterEstimate(descriptor.lhs, descriptor.operator, descriptor.rhs,
                        estimates[parameter], 0.0, Double.NaN, Double.NaN,
                        standardized[parameter], false));
                continue;
            }
            double[] gradient = reportedParameterGradient(
                    optimum, parameter, layout, specification, sampleStandardDeviations, sampleSize);
            double variance = quadratic(gradient, inverseHessian) * covarianceScale;
            double standardError = variance > 0 && Double.isFinite(variance)
                    ? Math.sqrt(variance) : Double.NaN;
            double z = standardError > 0 ? estimates[parameter] / standardError : Double.NaN;
            double p = Double.isFinite(z) ? normalTwoSidedP(z) : Double.NaN;
            result.add(new SemParameterEstimate(descriptor.lhs, descriptor.operator, descriptor.rhs,
                    estimates[parameter], standardError, z, p,
                    standardized[parameter], true));
        }
        return result;
    }

    private List<ParameterDescriptor> parameterDescriptors(CompositeSpecification specification,
                                                           String targetName) {
        List<ParameterDescriptor> result = new ArrayList<>();
        List<String> indicators = flattenedIndicators(specification);
        for (CompositeSpecification.Construct construct : specification.constructs()) {
            for (int local = 0; local < construct.indicators().size(); local++) {
                result.add(new ParameterDescriptor(
                        construct.name(), "=~", construct.indicators().get(local), local != 0));
            }
        }
        for (String indicator : indicators) {
            result.add(new ParameterDescriptor(indicator, "~~", indicator, true));
        }
        for (int row = 0; row < specification.constructs().size(); row++) {
            for (int column = 0; column <= row; column++) {
                result.add(new ParameterDescriptor(
                        specification.constructs().get(row).name(), "~~",
                        specification.constructs().get(column).name(), true));
            }
        }
        for (CompositeSpecification.Construct construct : specification.constructs()) {
            result.add(new ParameterDescriptor(targetName, "~", construct.name(), true));
        }
        result.add(new ParameterDescriptor(targetName, "~~", targetName, true));
        return result;
    }

    private double[] reportedParameterValues(
            double[] parameters,
            Layout layout,
            CompositeSpecification specification,
            double[] sampleStandardDeviations,
            int sampleSize,
            boolean standardized) {
        Decoded decoded = decode(parameters, layout);
        double[][] implied = impliedCovariance(decoded);
        double targetVariance = implied[layout.indicatorCount][layout.indicatorCount];
        double mlScale = Math.sqrt((sampleSize - 1.0) / sampleSize);
        double[] mlStandardDeviations = new double[sampleStandardDeviations.length];
        for (int index = 0; index < mlStandardDeviations.length; index++) {
            mlStandardDeviations[index] = sampleStandardDeviations[index] * mlScale;
        }
        int resultSize = 2 * layout.indicatorCount
                + layout.constructCount * (layout.constructCount + 1) / 2
                + layout.constructCount + 1;
        double[] result = new double[resultSize];
        int output = 0;
        int indicator = 0;
        for (int construct = 0; construct < layout.constructCount; construct++) {
            double markerScale = mlStandardDeviations[layout.firstIndicator[construct]];
            for (int local = 0; local < layout.indicatorCounts[construct]; local++) {
                if (standardized) {
                    result[output++] = decoded.lambda[indicator][construct]
                            * Math.sqrt(decoded.phi[construct][construct] / implied[indicator][indicator]);
                } else {
                    result[output++] = decoded.lambda[indicator][construct]
                            * mlStandardDeviations[indicator] / markerScale;
                }
                indicator++;
            }
        }
        for (indicator = 0; indicator < layout.indicatorCount; indicator++) {
            result[output++] = standardized
                    ? decoded.theta[indicator] / implied[indicator][indicator]
                    : decoded.theta[indicator] * mlStandardDeviations[indicator]
                    * mlStandardDeviations[indicator];
        }
        for (int row = 0; row < layout.constructCount; row++) {
            double rowScale = mlStandardDeviations[layout.firstIndicator[row]];
            for (int column = 0; column <= row; column++) {
                double columnScale = mlStandardDeviations[layout.firstIndicator[column]];
                result[output++] = standardized
                        ? decoded.phi[row][column]
                        / Math.sqrt(decoded.phi[row][row] * decoded.phi[column][column])
                        : decoded.phi[row][column] * rowScale * columnScale;
            }
        }
        double targetScale = mlStandardDeviations[layout.indicatorCount];
        for (int construct = 0; construct < layout.constructCount; construct++) {
            result[output++] = standardized
                    ? decoded.beta[construct]
                    * Math.sqrt(decoded.phi[construct][construct] / targetVariance)
                    : decoded.beta[construct] * targetScale
                    / mlStandardDeviations[layout.firstIndicator[construct]];
        }
        result[output] = standardized
                ? decoded.psi / targetVariance
                : decoded.psi * targetScale * targetScale;
        return result;
    }

    private double[] reportedParameterGradient(
            double[] optimum,
            int reportedParameter,
            Layout layout,
            CompositeSpecification specification,
            double[] sampleStandardDeviations,
            int sampleSize) {
        double[] result = new double[optimum.length];
        for (int parameter = 0; parameter < optimum.length; parameter++) {
            double step = 1.0e-5 * (1.0 + Math.abs(optimum[parameter]));
            double[] plus = optimum.clone();
            double[] minus = optimum.clone();
            plus[parameter] += step;
            minus[parameter] -= step;
            double plusValue = reportedParameterValues(
                    plus, layout, specification, sampleStandardDeviations, sampleSize, false)[reportedParameter];
            double minusValue = reportedParameterValues(
                    minus, layout, specification, sampleStandardDeviations, sampleSize, false)[reportedParameter];
            result[parameter] = (plusValue - minusValue) / (2.0 * step);
        }
        return result;
    }

    private double[][] inverseObservedHessian(ToDoubleFunction<double[]> objective, double[] optimum,
                                               double[][] fallbackInverseHessian) {
        double[][] hessian = numericalHessian(objective, optimum);
        double scale = 1.0;
        for (int index = 0; index < hessian.length; index++) {
            scale = Math.max(scale, Math.abs(hessian[index][index]));
        }
        double ridge = 0.0;
        for (int attempt = 0; attempt < 9; attempt++) {
            double[][] candidate = SemMatrixOps.copy(hessian);
            for (int index = 0; index < candidate.length; index++) candidate[index][index] += ridge;
            try {
                double[][] inverse = SemMatrixOps.inverse(candidate);
                boolean valid = true;
                for (int index = 0; index < inverse.length; index++) {
                    valid &= inverse[index][index] > 0 && Double.isFinite(inverse[index][index]);
                }
                if (valid) return inverse;
            } catch (IllegalArgumentException ignored) {
                // Increase the diagonal ridge and retry.
            }
            ridge = ridge == 0.0 ? scale * 1.0e-9 : ridge * 10.0;
        }
        return SemMatrixOps.copy(fallbackInverseHessian);
    }

    private double[][] numericalHessian(ToDoubleFunction<double[]> objective, double[] optimum) {
        int size = optimum.length;
        double[][] result = new double[size][size];
        double value = objective.applyAsDouble(optimum);
        double[] steps = new double[size];
        for (int index = 0; index < size; index++) {
            steps[index] = 1.0e-4 * (1.0 + Math.abs(optimum[index]));
            double[] plus = optimum.clone();
            double[] minus = optimum.clone();
            plus[index] += steps[index];
            minus[index] -= steps[index];
            result[index][index] = (objective.applyAsDouble(plus) - 2.0 * value
                    + objective.applyAsDouble(minus)) / (steps[index] * steps[index]);
        }
        for (int row = 0; row < size; row++) {
            for (int column = 0; column < row; column++) {
                double[] plusPlus = optimum.clone();
                double[] plusMinus = optimum.clone();
                double[] minusPlus = optimum.clone();
                double[] minusMinus = optimum.clone();
                plusPlus[row] += steps[row];
                plusPlus[column] += steps[column];
                plusMinus[row] += steps[row];
                plusMinus[column] -= steps[column];
                minusPlus[row] -= steps[row];
                minusPlus[column] += steps[column];
                minusMinus[row] -= steps[row];
                minusMinus[column] -= steps[column];
                double second = (objective.applyAsDouble(plusPlus)
                        - objective.applyAsDouble(plusMinus)
                        - objective.applyAsDouble(minusPlus)
                        + objective.applyAsDouble(minusMinus))
                        / (4.0 * steps[row] * steps[column]);
                result[row][column] = second;
                result[column][row] = second;
            }
        }
        return result;
    }

    private double normalTwoSidedP(double z) {
        return regularizedGammaQ(0.5, 0.5 * z * z);
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

    private record ParameterDescriptor(String lhs, String operator, String rhs, boolean free) {
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
