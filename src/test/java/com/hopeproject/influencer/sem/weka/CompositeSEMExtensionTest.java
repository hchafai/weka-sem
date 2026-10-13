package com.hopeproject.influencer.sem.weka;

import org.junit.jupiter.api.Test;
import weka.classifiers.Evaluation;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instances;
import weka.core.SerializationHelper;
import weka.filters.Filter;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.util.ArrayList;
import java.util.Random;

import static org.assertj.core.api.Assertions.assertThat;

class CompositeSEMExtensionTest {

    private static final String SPECIFICATION =
            "Activation=engagement,comments:2,sentiment;Reach=followers,views";

    @Test
    void parsesArbitraryConstructsAndNormalizesWeights() {
        CompositeSpecification specification = CompositeSpecification.parse(SPECIFICATION);

        assertThat(specification.constructs()).hasSize(2);
        assertThat(specification.constructs().get(0).name()).isEqualTo("Activation");
        assertThat(specification.constructs().get(0).indicators())
                .containsExactly("engagement", "comments", "sentiment");
        assertThat(specification.constructs().get(0).weights())
                .containsExactly(0.25, 0.5, 0.25);
    }

    @Test
    void filterLearnsOnFirstBatchAndAddsLatentAttributes() throws Exception {
        Instances training = dataset(90);
        CompositeSEMFilter filter = new CompositeSEMFilter();
        filter.setConstructs(SPECIFICATION);
        filter.setInputFormat(training);

        Instances transformed = Filter.useFilter(training, filter);

        assertThat(transformed.numInstances()).isEqualTo(training.numInstances());
        assertThat(transformed.attribute("sem_activation")).isNotNull();
        assertThat(transformed.attribute("sem_reach")).isNotNull();
        assertThat(transformed.attribute("sem_performance")).isNotNull();
        assertThat(transformed.classAttribute().name()).isEqualTo("net_value");
        assertThat(filter.getModel()).isNotNull();
        assertThat(filter.getModel().trainingRows()).isEqualTo(90);

        Instances scoringBatch = new Instances(training, 0);
        scoringBatch.add(training.instance(0));
        scoringBatch.instance(0).setClassMissing();
        Instances rescored = Filter.useFilter(scoringBatch, filter);
        assertThat(rescored).hasSize(1);
        assertThat(rescored.instance(0).value(rescored.attribute("sem_activation"))).isFinite();
        assertThat(filter.getModel().trainingRows()).isEqualTo(90);
    }

    @Test
    void metaRegressorSupportsCrossValidationAndSerialization() throws Exception {
        Instances data = dataset(120);
        CompositeSEMRegressor regressor = new CompositeSEMRegressor();
        regressor.setConstructs(SPECIFICATION);

        Evaluation evaluation = new Evaluation(data);
        evaluation.crossValidateModel(regressor, data, 5, new Random(7));
        assertThat(evaluation.rootMeanSquaredError()).isFinite().isLessThan(30.0);

        regressor.buildClassifier(data);
        double prediction = regressor.classifyInstance(data.instance(0));
        assertThat(prediction).isFinite();
        double[][] loadings = regressor.getModel().measurementLoadings();
        assertThat(loadings.length).isEqualTo(2);
        assertThat(loadings[0]).hasSize(3);
        assertThat(loadings[1]).hasSize(2);
        assertThat(regressor.getModel().structuralPaths()).hasSize(2);

        ByteArrayOutputStream output = new ByteArrayOutputStream();
        SerializationHelper.write(output, regressor);
        CompositeSEMRegressor restored = (CompositeSEMRegressor) SerializationHelper.read(
                new ByteArrayInputStream(output.toByteArray()));
        assertThat(restored.classifyInstance(data.instance(1))).isFinite();
    }

    @Test
    void missingIndicatorsAreMeanImputedInsideTheModel() throws Exception {
        Instances data = dataset(40);
        data.instance(3).setMissing(data.attribute("comments"));
        data.instance(7).setMissing(data.attribute("views"));

        CompositeSEMRegressor regressor = new CompositeSEMRegressor();
        regressor.setConstructs(SPECIFICATION);
        regressor.buildClassifier(data);

        assertThat(regressor.classifyInstance(data.instance(3))).isFinite();
        assertThat(regressor.getModel().srmr()).isFinite().isGreaterThanOrEqualTo(0.0);
    }

    private Instances dataset(int size) {
        ArrayList<Attribute> attributes = new ArrayList<>();
        attributes.add(new Attribute("engagement"));
        attributes.add(new Attribute("comments"));
        attributes.add(new Attribute("sentiment"));
        attributes.add(new Attribute("followers"));
        attributes.add(new Attribute("views"));
        attributes.add(new Attribute("net_value"));
        Instances data = new Instances("generic-composite-sem", attributes, size);
        data.setClassIndex(data.numAttributes() - 1);

        for (int i = 0; i < size; i++) {
            double engagement = 0.01 + (i % 17) * 0.004;
            double comments = 10 + (i * 7) % 43;
            double sentiment = -0.4 + (i % 11) * 0.08;
            double followers = 1_000 + i * 137;
            double views = 600 + i * 91 + (i % 5) * 80;
            double netValue = 450 * engagement + 1.8 * comments + 35 * sentiment
                    + 0.004 * followers + 0.006 * views + ((i % 3) - 1) * 0.15;
            DenseInstance instance = new DenseInstance(1.0,
                    new double[]{engagement, comments, sentiment, followers, views, netValue});
            instance.setDataset(data);
            data.add(instance);
        }
        return data;
    }
}
