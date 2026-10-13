package com.hopeproject.influencer.sem.weka;

import java.io.Serial;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.Locale;
import java.util.Set;

/**
 * Parser and immutable representation of a composite measurement model.
 *
 * <p>The command-line syntax is:</p>
 *
 * <pre>
 * Activation=engagement,comments:0.5,sentiment:0.5;Reach=followers,views
 * </pre>
 *
 * <p>An omitted weight defaults to 1. Weights are normalized by the sum of
 * their absolute values, which permits explicitly signed composites.</p>
 */
public final class CompositeSpecification implements Serializable {

    @Serial
    private static final long serialVersionUID = 1L;

    private final List<Construct> constructs;

    public CompositeSpecification(List<Construct> constructs) {
        if (constructs == null || constructs.isEmpty()) {
            throw new IllegalArgumentException("At least one construct is required");
        }
        Set<String> names = new HashSet<>();
        List<Construct> copy = new ArrayList<>();
        for (Construct construct : constructs) {
            String key = construct.name().toLowerCase(Locale.ROOT);
            if (!names.add(key)) {
                throw new IllegalArgumentException("Duplicate construct name: " + construct.name());
            }
            copy.add(construct);
        }
        this.constructs = Collections.unmodifiableList(copy);
    }

    public List<Construct> constructs() {
        return constructs;
    }

    public static CompositeSpecification parse(String source) {
        if (source == null || source.isBlank()) {
            throw new IllegalArgumentException(
                    "Composite specification is required, for example "
                            + "Activation=engagement,comments,sentiment;Reach=followers,views");
        }
        List<Construct> constructs = new ArrayList<>();
        for (String constructText : source.split(";")) {
            String trimmed = constructText.trim();
            if (trimmed.isEmpty()) continue;
            int separator = trimmed.indexOf('=');
            if (separator <= 0 || separator == trimmed.length() - 1) {
                throw new IllegalArgumentException("Invalid construct definition: " + trimmed);
            }
            String name = trimmed.substring(0, separator).trim();
            List<String> indicators = new ArrayList<>();
            List<Double> weights = new ArrayList<>();
            for (String indicatorText : trimmed.substring(separator + 1).split(",")) {
                String token = indicatorText.trim();
                if (token.isEmpty()) continue;
                String indicator = token;
                double weight = 1.0;
                int colon = token.lastIndexOf(':');
                if (colon > 0 && colon < token.length() - 1) {
                    String candidate = token.substring(colon + 1).trim();
                    try {
                        weight = Double.parseDouble(candidate);
                        indicator = token.substring(0, colon).trim();
                    } catch (NumberFormatException ignored) {
                        // A colon may be a legitimate part of an attribute name.
                    }
                }
                if (indicator.isEmpty() || !Double.isFinite(weight)) {
                    throw new IllegalArgumentException("Invalid indicator definition: " + token);
                }
                indicators.add(indicator);
                weights.add(weight);
            }
            constructs.add(new Construct(name, indicators, toArray(weights)));
        }
        return new CompositeSpecification(constructs);
    }

    private static double[] toArray(List<Double> values) {
        double[] result = new double[values.size()];
        for (int i = 0; i < values.size(); i++) result[i] = values.get(i);
        return result;
    }

    @Override
    public String toString() {
        List<String> definitions = new ArrayList<>();
        for (Construct construct : constructs) {
            List<String> indicators = new ArrayList<>();
            for (int i = 0; i < construct.indicators().size(); i++) {
                indicators.add(construct.indicators().get(i) + ":" + construct.weights()[i]);
            }
            definitions.add(construct.name() + "=" + String.join(",", indicators));
        }
        return String.join(";", definitions);
    }

    public static final class Construct implements Serializable {

        @Serial
        private static final long serialVersionUID = 1L;

        private final String name;
        private final List<String> indicators;
        private final double[] weights;

        public Construct(String name, List<String> indicators, double[] weights) {
            if (name == null || name.isBlank()) {
                throw new IllegalArgumentException("Construct name cannot be blank");
            }
            if (indicators == null || indicators.isEmpty()) {
                throw new IllegalArgumentException("Construct " + name + " must contain at least one indicator");
            }
            if (weights == null || weights.length != indicators.size()) {
                throw new IllegalArgumentException("Construct " + name + " has inconsistent weights");
            }
            Set<String> unique = new HashSet<>();
            List<String> indicatorCopy = new ArrayList<>();
            double absoluteTotal = 0;
            for (int i = 0; i < indicators.size(); i++) {
                String indicator = indicators.get(i);
                if (indicator == null || indicator.isBlank()) {
                    throw new IllegalArgumentException("Construct " + name + " contains a blank indicator");
                }
                if (!unique.add(indicator.toLowerCase(Locale.ROOT))) {
                    throw new IllegalArgumentException(
                            "Construct " + name + " contains duplicate indicator " + indicator);
                }
                if (!Double.isFinite(weights[i])) {
                    throw new IllegalArgumentException("Construct " + name + " contains a non-finite weight");
                }
                indicatorCopy.add(indicator);
                absoluteTotal += Math.abs(weights[i]);
            }
            if (absoluteTotal == 0) {
                throw new IllegalArgumentException("Construct " + name + " cannot have only zero weights");
            }
            this.name = name.trim();
            this.indicators = Collections.unmodifiableList(indicatorCopy);
            this.weights = new double[weights.length];
            for (int i = 0; i < weights.length; i++) this.weights[i] = weights[i] / absoluteTotal;
        }

        public String name() {
            return name;
        }

        public List<String> indicators() {
            return indicators;
        }

        public double[] weights() {
            return weights.clone();
        }
    }
}
