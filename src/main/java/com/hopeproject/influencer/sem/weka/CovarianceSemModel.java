package com.hopeproject.influencer.sem.weka;

import weka.core.Attribute;
import weka.core.Instance;

import java.io.Serial;
import java.io.Serializable;
import java.util.Arrays;

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
    private final double[] residualVariances;
    private final double[][] latentCovariance;
    private final double[] structuralPaths;
    private final double targetResidualVariance;
    private final double[][] factorScoreCoefficients;
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
            double[] residualVariances,
            double[][] latentCovariance,
            double[] structuralPaths,
            double targetResidualVariance,
            double[][] factorScoreCoefficients,
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
        this.residualVariances = residualVariances.clone();
        this.latentCovariance = SemMatrixOps.copy(latentCovariance);
        this.structuralPaths = structuralPaths.clone();
        this.targetResidualVariance = targetResidualVariance;
        this.factorScoreCoefficients = SemMatrixOps.copy(factorScoreCoefficients);
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

    public CompositeSpecification specification() { return specification; }
    public String[] indicatorNames() { return indicatorNames.clone(); }
    public double[] indicatorMeans() { return indicatorMeans.clone(); }
    public double[] indicatorStandardDeviations() { return indicatorStandardDeviations.clone(); }
    public double targetMean() { return targetMean; }
    public double targetStandardDeviation() { return targetStandardDeviation; }
    public double[][] loadings() { return SemMatrixOps.copy(loadings); }
    public double[] residualVariances() { return residualVariances.clone(); }
    public double[][] latentCovariance() { return SemMatrixOps.copy(latentCovariance); }
    public double[] structuralPaths() { return structuralPaths.clone(); }
    public double targetResidualVariance() { return targetResidualVariance; }
    public double chiSquare() { return chiSquare; }
    public int degreesOfFreedom() { return degreesOfFreedom; }
    public double chiSquarePValue() { return chiSquarePValue; }
    public double cfi() { return cfi; }
    public double tli() { return tli; }
    public double rmsea() { return rmsea; }
    public double srmr() { return srmr; }
    public double aic() { return aic; }
    public double bic() { return bic; }
    public double rSquared() { return rSquared; }
    public int trainingRows() { return trainingRows; }
    public int iterations() { return iterations; }
    public boolean converged() { return converged; }
    public double gradientNorm() { return gradientNorm; }
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
        result.append("Latent covariance:\n");
        for (double[] row : latentCovariance) result.append(Arrays.toString(row)).append('\n');
        for (int construct = 0; construct < specification.constructs().size(); construct++) {
            result.append(specification.constructs().get(construct).name())
                    .append(" -> class: beta=").append(structuralPaths[construct])
                    .append(", loadings=").append(Arrays.toString(loadings[construct])).append('\n');
        }
        return result.toString();
    }
}
