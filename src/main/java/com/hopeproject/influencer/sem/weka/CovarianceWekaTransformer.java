package com.hopeproject.influencer.sem.weka;

import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.Locale;
import java.util.Set;

/** Builds Weka datasets enriched with covariance-SEM factor scores. */
final class CovarianceWekaTransformer {

    private CovarianceWekaTransformer() {
    }

    static Instances outputFormat(Instances input, CompositeSpecification specification,
                                  String prefix, boolean latentOnly, boolean includePrediction) {
        if (input.classIndex() < 0) throw new IllegalArgumentException("A class attribute is required");
        String safePrefix = prefix == null ? "sem_ml_" : prefix;
        ArrayList<Attribute> attributes = new ArrayList<>();
        Set<String> names = new HashSet<>();
        if (!latentOnly) {
            for (int index = 0; index < input.numAttributes(); index++) {
                if (index == input.classIndex()) continue;
                Attribute source = input.attribute(index);
                attributes.add((Attribute) source.copy());
                names.add(source.name().toLowerCase(Locale.ROOT));
            }
        }
        for (CompositeSpecification.Construct construct : specification.constructs()) {
            String name = safePrefix + safeName(construct.name());
            ensureUnique(names, name);
            attributes.add(new Attribute(name));
        }
        if (includePrediction) {
            String name = safePrefix + "prediction";
            ensureUnique(names, name);
            attributes.add(new Attribute(name));
        }
        ensureUnique(names, input.classAttribute().name());
        attributes.add((Attribute) input.classAttribute().copy());
        Instances output = new Instances(input.relationName() + "-covariance-sem", attributes, 0);
        output.setClassIndex(output.numAttributes() - 1);
        return output;
    }

    static Instances transform(Instances input, CovarianceSemModel model, String prefix,
                               boolean latentOnly, boolean includePrediction) {
        Instances output = outputFormat(input, model.specification(), prefix, latentOnly, includePrediction);
        for (Instance instance : input) output.add(transform(instance, output, model, latentOnly, includePrediction));
        return output;
    }

    static Instance transform(Instance input, Instances outputHeader, CovarianceSemModel model,
                              boolean latentOnly, boolean includePrediction) {
        CovarianceSemScores scores = model.score(input);
        double[] values = new double[outputHeader.numAttributes()];
        int cursor = 0;
        if (!latentOnly) {
            for (int index = 0; index < input.numAttributes(); index++) {
                if (index == input.classIndex()) continue;
                values[cursor++] = input.value(index);
            }
        }
        for (double score : scores.standardizedFactorScores()) values[cursor++] = score;
        if (includePrediction) values[cursor++] = scores.prediction();
        values[cursor] = input.classIsMissing() ? Double.NaN : input.classValue();
        DenseInstance transformed = new DenseInstance(input.weight(), values);
        transformed.setDataset(outputHeader);
        return transformed;
    }

    private static String safeName(String source) {
        String result = source.trim().toLowerCase(Locale.ROOT).replaceAll("[^a-z0-9_]+", "_");
        result = result.replaceAll("^_+|_+$", "");
        return result.isBlank() ? "construct" : result;
    }

    private static void ensureUnique(Set<String> names, String name) {
        if (!names.add(name.toLowerCase(Locale.ROOT))) {
            throw new IllegalArgumentException("Duplicate output attribute name: " + name);
        }
    }
}
