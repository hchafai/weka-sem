package com.hopeproject.influencer.sem.weka;

import java.io.Serial;
import java.io.Serializable;

/** Scores produced for one Weka instance by a fitted composite model. */
public record CompositeScores(
        String[] constructNames,
        double[] constructZScores,
        double[] constructScores,
        double performanceZ,
        double performanceScore,
        double[] contributions
) implements Serializable {

    @Serial
    private static final long serialVersionUID = 1L;

    public CompositeScores {
        constructNames = constructNames.clone();
        constructZScores = constructZScores.clone();
        constructScores = constructScores.clone();
        contributions = contributions.clone();
    }
}
