package com.hopeproject.influencer.sem.weka;

import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.Locale;
import java.util.Set;

/** Converts Weka data to an augmented or latent-only representation. */
final class CompositeWekaTransformer {

    private CompositeWekaTransformer() {
    }

    static Instances outputFormat(Instances input, CompositeSpecification specification,
                                  String prefix, boolean latentOnly, boolean includePerformance) {
        if (input.classIndex() < 0) throw new IllegalArgumentException("A class attribute is required");
        String safePrefix = prefix == null ? "sem_" : prefix;
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
        if (includePerformance) {
            String name = safePrefix + "performance";
            ensureUnique(names, name);
            attributes.add(new Attribute(name));
        }
        String className = input.classAttribute().name();
        ensureUnique(names, className);
        attributes.add((Attribute) input.classAttribute().copy());
        Instances output = new Instances(input.relationName() + "-composite-sem", attributes, 0);
        output.setClassIndex(output.numAttributes() - 1);
        return output;
    }

    static Instances transform(Instances input, CompositeSemModel model, String prefix,
                               boolean latentOnly, boolean includePerformance) {
        Instances output = outputFormat(input, model.specification(), prefix, latentOnly, includePerformance);
        for (Instance instance : input) output.add(transform(instance, output, model, latentOnly, includePerformance));
        return output;
    }

    static Instance transform(Instance input, Instances outputHeader, CompositeSemModel model,
                              boolean latentOnly, boolean includePerformance) {
        CompositeScores scores = model.score(input);
        double[] values = new double[outputHeader.numAttributes()];
        int cursor = 0;
        if (!latentOnly) {
            for (int index = 0; index < input.numAttributes(); index++) {
                if (index == input.classIndex()) continue;
                values[cursor++] = input.value(index);
            }
        }
        for (double score : scores.constructZScores()) values[cursor++] = score;
        if (includePerformance) values[cursor++] = scores.performanceZ();
        values[cursor] = input.classIsMissing() ? Double.NaN : input.classValue();
        DenseInstance transformed = new DenseInstance(input.weight(), values);
        transformed.setDataset(outputHeader);
        return transformed;
    }

    private static String safeName(String source) {
        String normalized = source.trim().toLowerCase(Locale.ROOT).replaceAll("[^a-z0-9_]+", "_");
        normalized = normalized.replaceAll("^_+|_+$", "");
        return normalized.isBlank() ? "construct" : normalized;
    }

    private static void ensureUnique(Set<String> names, String name) {
        if (!names.add(name.toLowerCase(Locale.ROOT))) {
            throw new IllegalArgumentException("Duplicate output attribute name: " + name);
        }
    }
}
