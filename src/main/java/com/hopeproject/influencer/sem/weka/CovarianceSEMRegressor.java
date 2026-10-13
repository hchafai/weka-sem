package com.hopeproject.influencer.sem.weka;

import weka.classifiers.AbstractClassifier;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Option;
import weka.core.TechnicalInformation;
import weka.core.TechnicalInformationHandler;
import weka.core.Utils;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Enumeration;
import java.util.List;

/** Weka regressor backed by maximum-likelihood covariance SEM. */
public class CovarianceSEMRegressor extends AbstractClassifier implements TechnicalInformationHandler {

    private static final long serialVersionUID = 1L;

    private String constructs = "";
    private double regularization = 1.0e-6;
    private int maximumIterations = 300;
    private double tolerance = 1.0e-6;
    private CovarianceSemModel model;

    public String globalInfo() {
        return "Maximum-likelihood covariance SEM for reflective latent predictors and one observed "
                + "numeric outcome. Marker loadings identify the factors; all other loadings, "
                + "measurement-error variances, latent covariances, paths and outcome residual "
                + "variance are jointly estimated from the covariance matrix.\n\n" + getTechnicalInformation();
    }

    @Override
    public void buildClassifier(Instances source) throws Exception {
        getCapabilities().testWithFail(source);
        if (constructs.isBlank()) throw new IllegalArgumentException("Set -constructs before training");
        Instances training = new Instances(source);
        training.deleteWithMissingClass();
        model = new CovarianceSemEstimator().fit(training, CompositeSpecification.parse(constructs),
                regularization, maximumIterations, tolerance);
    }

    @Override
    public double classifyInstance(Instance instance) {
        if (model == null) throw new IllegalStateException("CovarianceSEMRegressor has not been built yet");
        return model.score(instance).prediction();
    }

    @Override
    public Capabilities getCapabilities() {
        Capabilities capabilities = super.getCapabilities();
        capabilities.disableAll();
        capabilities.enable(Capabilities.Capability.NUMERIC_ATTRIBUTES);
        capabilities.enable(Capabilities.Capability.MISSING_VALUES);
        capabilities.enable(Capabilities.Capability.NUMERIC_CLASS);
        capabilities.enable(Capabilities.Capability.MISSING_CLASS_VALUES);
        capabilities.setMinimumNumberInstances(6);
        return capabilities;
    }

    @Override
    public Enumeration<Option> listOptions() {
        List<Option> options = new ArrayList<>();
        options.add(new Option("Reflective measurement model: Factor=a,b,c;Other=d,e,f",
                "constructs", 1, "-constructs <specification>"));
        options.add(new Option("Positive diagonal regularization for the sample covariance matrix.",
                "regularization", 1, "-regularization <double>"));
        options.add(new Option("Maximum BFGS iterations (default 300).",
                "max-iterations", 1, "-max-iterations <integer>"));
        options.add(new Option("Optimization tolerance (default 1.0E-6).",
                "tolerance", 1, "-tolerance <double>"));
        return Collections.enumeration(options);
    }

    @Override
    public void setOptions(String[] options) throws Exception {
        String value = Utils.getOption("constructs", options);
        if (!value.isBlank()) setConstructs(value);
        value = Utils.getOption("regularization", options);
        if (!value.isBlank()) setRegularization(Double.parseDouble(value));
        value = Utils.getOption("max-iterations", options);
        if (!value.isBlank()) setMaximumIterations(Integer.parseInt(value));
        value = Utils.getOption("tolerance", options);
        if (!value.isBlank()) setTolerance(Double.parseDouble(value));
        Utils.checkForRemainingOptions(options);
    }

    @Override
    public String[] getOptions() {
        List<String> options = new ArrayList<>();
        if (!constructs.isBlank()) Collections.addAll(options, "-constructs", constructs);
        Collections.addAll(options,
                "-regularization", Double.toString(regularization),
                "-max-iterations", Integer.toString(maximumIterations),
                "-tolerance", Double.toString(tolerance));
        return options.toArray(String[]::new);
    }

    @Override
    public TechnicalInformation getTechnicalInformation() {
        TechnicalInformation information = new TechnicalInformation(TechnicalInformation.Type.BOOK);
        information.setValue(TechnicalInformation.Field.AUTHOR, "Bollen, K. A.");
        information.setValue(TechnicalInformation.Field.YEAR, "1989");
        information.setValue(TechnicalInformation.Field.TITLE, "Structural Equations with Latent Variables");
        information.setValue(TechnicalInformation.Field.PUBLISHER, "Wiley");
        information.setValue(TechnicalInformation.Field.ISBN, "978-0-471-01171-2");
        return information;
    }

    public String constructsTipText() { return "Reflective measurement-model specification."; }
    public String getConstructs() { return constructs; }
    public void setConstructs(String constructs) { this.constructs = constructs == null ? "" : constructs.trim(); }
    public String regularizationTipText() { return "Regularization added to the sample covariance diagonal."; }
    public double getRegularization() { return regularization; }
    public void setRegularization(double regularization) {
        if (!Double.isFinite(regularization) || regularization <= 0) {
            throw new IllegalArgumentException("Regularization must be strictly positive");
        }
        this.regularization = regularization;
    }
    public String maximumIterationsTipText() { return "Maximum number of BFGS iterations."; }
    public int getMaximumIterations() { return maximumIterations; }
    public void setMaximumIterations(int maximumIterations) {
        if (maximumIterations < 1) throw new IllegalArgumentException("Maximum iterations must be positive");
        this.maximumIterations = maximumIterations;
    }
    public String toleranceTipText() { return "Convergence tolerance for the numerical optimizer."; }
    public double getTolerance() { return tolerance; }
    public void setTolerance(double tolerance) {
        if (!Double.isFinite(tolerance) || tolerance <= 0) {
            throw new IllegalArgumentException("Tolerance must be strictly positive");
        }
        this.tolerance = tolerance;
    }
    public CovarianceSemModel getModel() { return model; }

    @Override
    public String toString() {
        return model == null ? "CovarianceSEMRegressor: no model built yet." : model.toString();
    }

    @Override
    public String getRevision() { return "1.0"; }

    public static void main(String[] args) {
        runClassifier(new CovarianceSEMRegressor(), args);
    }
}
