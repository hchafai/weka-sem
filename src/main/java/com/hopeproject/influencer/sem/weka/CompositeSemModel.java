package com.hopeproject.influencer.sem.weka;

import weka.core.Attribute;
import weka.core.Instance;

import java.io.Serial;
import java.io.Serializable;
import java.util.Arrays;

/** Immutable fitted composite path model. */
public final class CompositeSemModel implements Serializable {

    @Serial
    private static final long serialVersionUID = 1L;

    private static final double EPSILON = 1.0e-12;

    private final CompositeSpecification specification;
    private final double[][] indicatorMeans;
    private final double[][] indicatorStandardDeviations;
    private final double[] constructMeans;
    private final double[] constructStandardDeviations;
    private final double[][] measurementLoadings;
    private final double[] compositeReliability;
    private final double[] structuralPaths;
    private final double[] standardErrors;
    private final double[] pValues;
    private final double targetMean;
    private final double targetStandardDeviation;
    private final double rSquared;
    private final double adjustedRSquared;
    private final double structuralRmse;
    private final double srmr;
    private final int trainingRows;
    private final double ridge;

    CompositeSemModel(
            CompositeSpecification specification,
            double[][] indicatorMeans,
            double[][] indicatorStandardDeviations,
            double[] constructMeans,
            double[] constructStandardDeviations,
            double[][] measurementLoadings,
            double[] compositeReliability,
            double[] structuralPaths,
            double[] standardErrors,
            double[] pValues,
            double targetMean,
            double targetStandardDeviation,
            double rSquared,
            double adjustedRSquared,
            double structuralRmse,
            double srmr,
            int trainingRows,
            double ridge
    ) {
        this.specification = specification;
        this.indicatorMeans = deepCopy(indicatorMeans);
        this.indicatorStandardDeviations = deepCopy(indicatorStandardDeviations);
        this.constructMeans = constructMeans.clone();
        this.constructStandardDeviations = constructStandardDeviations.clone();
        this.measurementLoadings = deepCopy(measurementLoadings);
        this.compositeReliability = compositeReliability.clone();
        this.structuralPaths = structuralPaths.clone();
        this.standardErrors = standardErrors.clone();
        this.pValues = pValues.clone();
        this.targetMean = targetMean;
        this.targetStandardDeviation = targetStandardDeviation;
        this.rSquared = rSquared;
        this.adjustedRSquared = adjustedRSquared;
        this.structuralRmse = structuralRmse;
        this.srmr = srmr;
        this.trainingRows = trainingRows;
        this.ridge = ridge;
    }

    public CompositeScores score(Instance instance) {
        if (instance.dataset() == null) {
            throw new IllegalArgumentException("The instance must be attached to a Weka dataset header");
        }
        int constructCount = specification.constructs().size();
        double[] constructZ = new double[constructCount];
        double[] normalized = new double[constructCount];
        double[] contributions = new double[constructCount];
        String[] names = new String[constructCount];
        double performance = 0;

        for (int constructIndex = 0; constructIndex < constructCount; constructIndex++) {
            CompositeSpecification.Construct construct = specification.constructs().get(constructIndex);
            names[constructIndex] = construct.name();
            double[] weights = construct.weights();
            double raw = 0;
            for (int indicatorIndex = 0; indicatorIndex < construct.indicators().size(); indicatorIndex++) {
                String name = construct.indicators().get(indicatorIndex);
                Attribute attribute = instance.dataset().attribute(name);
                if (attribute == null) {
                    throw new IllegalArgumentException("Missing indicator attribute in scoring data: " + name);
                }
                if (!attribute.isNumeric()) {
                    throw new IllegalArgumentException("Indicator attribute must be numeric: " + name);
                }
                double z = instance.isMissing(attribute)
                        ? 0.0
                        : (instance.value(attribute) - indicatorMeans[constructIndex][indicatorIndex])
                        / safeStd(indicatorStandardDeviations[constructIndex][indicatorIndex]);
                raw += weights[indicatorIndex] * z;
            }
            constructZ[constructIndex] = (raw - constructMeans[constructIndex])
                    / safeStd(constructStandardDeviations[constructIndex]);
            normalized[constructIndex] = normalizedScore(constructZ[constructIndex]);
            contributions[constructIndex] = constructZ[constructIndex] * structuralPaths[constructIndex];
            performance += contributions[constructIndex];
        }
        return new CompositeScores(names, constructZ, normalized, performance,
                normalizedScore(performance), contributions);
    }

    private double safeStd(double value) {
        return Math.abs(value) < EPSILON ? 1.0 : value;
    }

    private double normalizedScore(double z) {
        return Math.round((100.0 / (1.0 + Math.exp(-z))) * 100.0) / 100.0;
    }

    private static double[][] deepCopy(double[][] values) {
        double[][] copy = new double[values.length][];
        for (int i = 0; i < values.length; i++) copy[i] = values[i].clone();
        return copy;
    }

    public CompositeSpecification specification() { return specification; }
    public double[][] indicatorMeans() { return deepCopy(indicatorMeans); }
    public double[][] indicatorStandardDeviations() { return deepCopy(indicatorStandardDeviations); }
    public double[] constructMeans() { return constructMeans.clone(); }
    public double[] constructStandardDeviations() { return constructStandardDeviations.clone(); }
    public double[][] measurementLoadings() { return deepCopy(measurementLoadings); }
    public double[] compositeReliability() { return compositeReliability.clone(); }
    public double[] structuralPaths() { return structuralPaths.clone(); }
    public double[] standardErrors() { return standardErrors.clone(); }
    public double[] pValues() { return pValues.clone(); }
    public double targetMean() { return targetMean; }
    public double targetStandardDeviation() { return targetStandardDeviation; }
    public double rSquared() { return rSquared; }
    public double adjustedRSquared() { return adjustedRSquared; }
    public double structuralRmse() { return structuralRmse; }
    public double srmr() { return srmr; }
    public int trainingRows() { return trainingRows; }
    public double ridge() { return ridge; }

    @Override
    public String toString() {
        StringBuilder result = new StringBuilder("Composite path model\n");
        result.append("Training rows: ").append(trainingRows).append('\n');
        result.append("R^2: ").append(rSquared).append('\n');
        result.append("Adjusted R^2: ").append(adjustedRSquared).append('\n');
        result.append("Structural RMSE: ").append(structuralRmse).append('\n');
        result.append("Approximate SRMR: ").append(srmr).append('\n');
        for (int i = 0; i < specification.constructs().size(); i++) {
            result.append(specification.constructs().get(i).name())
                    .append(" -> class: beta=").append(structuralPaths[i])
                    .append(", SE=").append(standardErrors[i])
                    .append(", p=").append(pValues[i])
                    .append(", rho_c=").append(compositeReliability[i])
                    .append(", loadings=").append(Arrays.toString(measurementLoadings[i]))
                    .append('\n');
        }
        return result.toString();
    }
}
