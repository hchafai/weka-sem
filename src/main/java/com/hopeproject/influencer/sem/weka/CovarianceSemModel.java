package com.hopeproject.influencer.sem.weka;

import weka.core.Attribute;
import weka.core.Instance;

import java.io.Serial;
import java.io.Serializable;
import java.util.Arrays;
import java.util.List;
import java.util.Locale;

/** Immutable fitted covariance-based SEM with an observed numeric outcome. */
public final class CovarianceSemModel implements Serializable {

    @Serial
    private static final long serialVersionUID = 1L;

    private final CompositeSpecification specification;
    private final String[] indicatorNames;
    private final double[] indicatorMeans;
    private final double[] indicatorStandardDeviations;
    private final double targetMean;
    private final double targetStandardDeviation;
    private final double[][] loadings;
    private final double[][] standardizedLoadings;
    private final double[] residualVariances;
    private final double[] standardizedResidualVariances;
    private final double[][] latentCovariance;
    private final double[][] latentCorrelations;
    private final double[] structuralPaths;
    private final double[] standardizedStructuralPaths;
    private final double targetResidualVariance;
    private final double standardizedTargetResidualVariance;
    private final double[][] factorScoreCoefficients;
    private final List<SemParameterEstimate> parameterEstimates;
    private final double chiSquare;
    private final int degreesOfFreedom;
    private final double chiSquarePValue;
    private final double cfi;
    private final double tli;
    private final double rmsea;
    private final double srmr;
    private final double aic;
    private final double bic;
    private final double rSquared;
    private final int trainingRows;
    private final int iterations;
    private final boolean converged;
    private final double gradientNorm;
    private final double regularization;

    CovarianceSemModel(
            CompositeSpecification specification,
            String[] indicatorNames,
            double[] indicatorMeans,
            double[] indicatorStandardDeviations,
            double targetMean,
            double targetStandardDeviation,
            double[][] loadings,
            double[][] standardizedLoadings,
            double[] residualVariances,
            double[] standardizedResidualVariances,
            double[][] latentCovariance,
            double[][] latentCorrelations,
            double[] structuralPaths,
            double[] standardizedStructuralPaths,
            double targetResidualVariance,
            double standardizedTargetResidualVariance,
            double[][] factorScoreCoefficients,
            List<SemParameterEstimate> parameterEstimates,
            double chiSquare,
            int degreesOfFreedom,
            double chiSquarePValue,
            double cfi,
            double tli,
            double rmsea,
            double srmr,
            double aic,
            double bic,
            double rSquared,
            int trainingRows,
            int iterations,
            boolean converged,
            double gradientNorm,
            double regularization
    ) {
        this.specification = specification;
        this.indicatorNames = indicatorNames.clone();
        this.indicatorMeans = indicatorMeans.clone();
        this.indicatorStandardDeviations = indicatorStandardDeviations.clone();
        this.targetMean = targetMean;
        this.targetStandardDeviation = targetStandardDeviation;
        this.loadings = SemMatrixOps.copy(loadings);
        this.standardizedLoadings = SemMatrixOps.copy(standardizedLoadings);
        this.residualVariances = residualVariances.clone();
        this.standardizedResidualVariances = standardizedResidualVariances.clone();
        this.latentCovariance = SemMatrixOps.copy(latentCovariance);
        this.latentCorrelations = SemMatrixOps.copy(latentCorrelations);
        this.structuralPaths = structuralPaths.clone();
        this.standardizedStructuralPaths = standardizedStructuralPaths.clone();
        this.targetResidualVariance = targetResidualVariance;
        this.standardizedTargetResidualVariance = standardizedTargetResidualVariance;
        this.factorScoreCoefficients = SemMatrixOps.copy(factorScoreCoefficients);
        this.parameterEstimates = List.copyOf(parameterEstimates);
        this.chiSquare = chiSquare;
        this.degreesOfFreedom = degreesOfFreedom;
        this.chiSquarePValue = chiSquarePValue;
        this.cfi = cfi;
        this.tli = tli;
        this.rmsea = rmsea;
        this.srmr = srmr;
        this.aic = aic;
        this.bic = bic;
        this.rSquared = rSquared;
        this.trainingRows = trainingRows;
        this.iterations = iterations;
        this.converged = converged;
        this.gradientNorm = gradientNorm;
        this.regularization = regularization;
    }

    /** Scores one instance without consulting its class value. */
    public CovarianceSemScores score(Instance instance) {
        if (instance.dataset() == null) {
            throw new IllegalArgumentException("The instance must be attached to a Weka dataset header");
        }
        double[] standardizedIndicators = new double[indicatorNames.length];
        for (int index = 0; index < indicatorNames.length; index++) {
            Attribute attribute = instance.dataset().attribute(indicatorNames[index]);
            if (attribute == null || !attribute.isNumeric()) {
                throw new IllegalArgumentException("Missing numeric indicator attribute: " + indicatorNames[index]);
            }
            standardizedIndicators[index] = instance.isMissing(attribute)
                    ? 0.0
                    : (instance.value(attribute) - indicatorMeans[index]) / indicatorStandardDeviations[index];
        }
        double[] factors = SemMatrixOps.multiply(factorScoreCoefficients, standardizedIndicators);
        double[] standardizedFactors = new double[factors.length];
        String[] names = new String[factors.length];
        for (int index = 0; index < factors.length; index++) {
            standardizedFactors[index] = factors[index] / Math.sqrt(latentCovariance[index][index]);
            names[index] = specification.constructs().get(index).name();
        }
        double standardizedPrediction = SemMatrixOps.dot(structuralPaths, factors);
        return new CovarianceSemScores(names, factors, standardizedFactors,
                standardizedPrediction, targetMean + targetStandardDeviation * standardizedPrediction);
    }

    /** Returns the fitted measurement specification. */
    public CompositeSpecification specification() { return specification; }
    /** Returns indicator names in covariance-matrix order. */
    public String[] indicatorNames() { return indicatorNames.clone(); }
    /** Returns training means used for scoring indicators. */
    public double[] indicatorMeans() { return indicatorMeans.clone(); }
    /** Returns training sample standard deviations used for scoring indicators. */
    public double[] indicatorStandardDeviations() { return indicatorStandardDeviations.clone(); }
    /** Returns the training target mean. */
    public double targetMean() { return targetMean; }
    /** Returns the training target sample standard deviation. */
    public double targetStandardDeviation() { return targetStandardDeviation; }
    /** Returns marker-identified loadings in the standardized-input parameterization. */
    public double[][] loadings() { return SemMatrixOps.copy(loadings); }
    /** Returns completely standardized factor loadings. */
    public double[][] standardizedLoadings() { return SemMatrixOps.copy(standardizedLoadings); }
    /** Returns indicator residual variances in the standardized-input parameterization. */
    public double[] residualVariances() { return residualVariances.clone(); }
    /** Returns completely standardized indicator residual variances. */
    public double[] standardizedResidualVariances() { return standardizedResidualVariances.clone(); }
    /** Returns latent covariances in the standardized-input parameterization. */
    public double[][] latentCovariance() { return SemMatrixOps.copy(latentCovariance); }
    /** Returns the latent correlation matrix. */
    public double[][] latentCorrelations() { return SemMatrixOps.copy(latentCorrelations); }
    /** Returns paths in the standardized-input parameterization. */
    public double[] structuralPaths() { return structuralPaths.clone(); }
    /** Returns completely standardized structural paths. */
    public double[] standardizedStructuralPaths() { return standardizedStructuralPaths.clone(); }
    /** Returns target residual variance in the standardized-input parameterization. */
    public double targetResidualVariance() { return targetResidualVariance; }
    /** Returns completely standardized target residual variance. */
    public double standardizedTargetResidualVariance() { return standardizedTargetResidualVariance; }
    /** Returns original-scale estimates, observed-information inference, and std.all estimates. */
    public List<SemParameterEstimate> parameterEstimates() { return parameterEstimates; }
    /** Returns the normal-theory likelihood-ratio chi-square. */
    public double chiSquare() { return chiSquare; }
    /** Returns model degrees of freedom. */
    public int degreesOfFreedom() { return degreesOfFreedom; }
    /** Returns the chi-square survival probability. */
    public double chiSquarePValue() { return chiSquarePValue; }
    /** Returns the comparative fit index. */
    public double cfi() { return cfi; }
    /** Returns the Tucker-Lewis index. */
    public double tli() { return tli; }
    /** Returns the root mean square error of approximation. */
    public double rmsea() { return rmsea; }
    /** Returns the standardized root mean square residual. */
    public double srmr() { return srmr; }
    /** Returns full-likelihood Akaike information criterion. */
    public double aic() { return aic; }
    /** Returns full-likelihood Bayesian information criterion. */
    public double bic() { return bic; }
    /** Returns explained variance of the observed numeric target. */
    public double rSquared() { return rSquared; }
    /** Returns the number of complete-target training rows. */
    public int trainingRows() { return trainingRows; }
    /** Returns optimizer iterations. */
    public int iterations() { return iterations; }
    /** Returns whether the numerical convergence rule was satisfied. */
    public boolean converged() { return converged; }
    /** Returns the infinity norm of the final discrepancy gradient. */
    public double gradientNorm() { return gradientNorm; }
    /** Returns the covariance ridge used during parameter estimation. */
    public double regularization() { return regularization; }

    @Override
    public String toString() {
        StringBuilder result = new StringBuilder("Covariance SEM (maximum likelihood)\n");
        result.append("Training rows: ").append(trainingRows).append('\n');
        result.append("Converged: ").append(converged)
                .append(" after ").append(iterations).append(" iterations")
                .append("; gradient infinity norm=").append(gradientNorm).append('\n');
        result.append("Chi-square(").append(degreesOfFreedom).append(")=").append(chiSquare)
                .append(", p=").append(chiSquarePValue).append('\n');
        result.append("CFI=").append(cfi).append(", TLI=").append(tli)
                .append(", RMSEA=").append(rmsea).append(", SRMR=").append(srmr).append('\n');
        result.append("AIC=").append(aic).append(", BIC=").append(bic)
                .append(", target R^2=").append(rSquared).append('\n');
        result.append("Latent covariance (standardized-input parameterization):\n");
        for (double[] row : latentCovariance) result.append(Arrays.toString(row)).append('\n');
        for (int construct = 0; construct < specification.constructs().size(); construct++) {
            result.append(specification.constructs().get(construct).name())
                    .append(" -> class: beta=").append(structuralPaths[construct])
                    .append(", loadings=").append(Arrays.toString(loadings[construct])).append('\n');
        }
        result.append("Fully standardized solution (std.all):\n");
        result.append(String.format(Locale.ROOT, "%-26s %12s %12s %10s %12s %12s%n",
                "parameter", "estimate", "std.error", "z", "p", "std.all"));
        for (SemParameterEstimate parameter : parameterEstimates) {
            result.append(String.format(Locale.ROOT, "%-26s %12.6f %12.6f %10.3f %12.6g %12.6f%n",
                    parameter.label(), parameter.estimate(), parameter.standardError(),
                    parameter.zStatistic(), parameter.pValue(), parameter.standardizedEstimate()));
        }
        if (parameterEstimates.isEmpty()) {
            result.append("Parameter inference was disabled for this fit.\n");
        }
        return result.toString();
    }
}
