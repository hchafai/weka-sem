package com.hopeproject.influencer.sem.weka;

import weka.core.Capabilities;
import weka.core.Instances;
import weka.core.Option;
import weka.core.Utils;
import weka.filters.SimpleBatchFilter;
import weka.filters.SupervisedFilter;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Enumeration;
import java.util.List;

/** Supervised filter that appends factor scores from a fitted covariance SEM. */
public class CovarianceSEMFilter extends SimpleBatchFilter implements SupervisedFilter {

    private static final long serialVersionUID = 1L;

    private String constructs = "";
    private double regularization = 1.0e-6;
    private int maximumIterations = 300;
    private double tolerance = 1.0e-6;
    private String prefix = "sem_ml_";
    private boolean latentOnly;
    private boolean includePrediction = true;
    private CovarianceSemModel model;

    @Override
    public String globalInfo() {
        return "Fits a maximum-likelihood covariance SEM on the first supervised batch and appends "
                + "regression factor scores plus the SEM prediction. The fitted model is reused "
                + "unchanged for subsequent batches.";
    }

    @Override
    protected Instances determineOutputFormat(Instances inputFormat) {
        return CovarianceWekaTransformer.outputFormat(inputFormat,
                CompositeSpecification.parse(constructs), prefix, latentOnly, includePrediction);
    }

    @Override
    protected Instances process(Instances instances) {
        if (model == null) {
            model = new CovarianceSemEstimator().fit(instances, CompositeSpecification.parse(constructs),
                    regularization, maximumIterations, tolerance);
        }
        return CovarianceWekaTransformer.transform(instances, model, prefix, latentOnly, includePrediction);
    }

    @Override
    protected void reset() {
        super.reset();
        model = null;
    }

    @Override
    public Capabilities getCapabilities() {
        Capabilities capabilities = super.getCapabilities();
        capabilities.disableAll();
        capabilities.enable(Capabilities.Capability.NUMERIC_ATTRIBUTES);
        capabilities.enable(Capabilities.Capability.MISSING_VALUES);
        capabilities.enable(Capabilities.Capability.NUMERIC_CLASS);
        capabilities.enable(Capabilities.Capability.MISSING_CLASS_VALUES);
        capabilities.setMinimumNumberInstances(0);
        return capabilities;
    }

    @Override
    public Enumeration<Option> listOptions() {
        List<Option> options = new ArrayList<>();
        options.add(new Option("Reflective measurement model: Factor=a,b,c;Other=d,e,f",
                "constructs", 1, "-constructs <specification>"));
        options.add(new Option("Positive diagonal regularization.",
                "regularization", 1, "-regularization <double>"));
        options.add(new Option("Maximum BFGS iterations.",
                "max-iterations", 1, "-max-iterations <integer>"));
        options.add(new Option("Optimization tolerance.",
                "tolerance", 1, "-tolerance <double>"));
        options.add(new Option("Prefix for generated attributes.", "prefix", 1, "-prefix <text>"));
        options.add(new Option("Output factor scores only, plus the class.",
                "latent-only", 0, "-latent-only"));
        options.add(new Option("Do not append the SEM prediction.",
                "no-prediction", 0, "-no-prediction"));
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
        value = Utils.getOption("prefix", options);
        if (!value.isBlank()) setPrefix(value);
        setLatentOnly(Utils.getFlag("latent-only", options));
        setIncludePrediction(!Utils.getFlag("no-prediction", options));
        Utils.checkForRemainingOptions(options);
    }

    @Override
    public String[] getOptions() {
        List<String> options = new ArrayList<>();
        if (!constructs.isBlank()) Collections.addAll(options, "-constructs", constructs);
        Collections.addAll(options,
                "-regularization", Double.toString(regularization),
                "-max-iterations", Integer.toString(maximumIterations),
                "-tolerance", Double.toString(tolerance),
                "-prefix", prefix);
        if (latentOnly) options.add("-latent-only");
        if (!includePrediction) options.add("-no-prediction");
        return options.toArray(String[]::new);
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
    public String maximumIterationsTipText() { return "Maximum BFGS iterations."; }
    public int getMaximumIterations() { return maximumIterations; }
    public void setMaximumIterations(int maximumIterations) {
        if (maximumIterations < 1) throw new IllegalArgumentException("Maximum iterations must be positive");
        this.maximumIterations = maximumIterations;
    }
    public String toleranceTipText() { return "Numerical convergence tolerance."; }
    public double getTolerance() { return tolerance; }
    public void setTolerance(double tolerance) {
        if (!Double.isFinite(tolerance) || tolerance <= 0) {
            throw new IllegalArgumentException("Tolerance must be strictly positive");
        }
        this.tolerance = tolerance;
    }
    public String prefixTipText() { return "Prefix for generated factor-score attributes."; }
    public String getPrefix() { return prefix; }
    public void setPrefix(String prefix) { this.prefix = prefix == null ? "sem_ml_" : prefix; }
    public String latentOnlyTipText() { return "Whether to retain only generated scores and the class."; }
    public boolean getLatentOnly() { return latentOnly; }
    public void setLatentOnly(boolean latentOnly) { this.latentOnly = latentOnly; }
    public String includePredictionTipText() { return "Whether to append the SEM outcome prediction."; }
    public boolean getIncludePrediction() { return includePrediction; }
    public void setIncludePrediction(boolean includePrediction) { this.includePrediction = includePrediction; }
    public CovarianceSemModel getModel() { return model; }

    @Override
    public String getRevision() { return "1.0"; }

    public static void main(String[] args) {
        runFilter(new CovarianceSEMFilter(), args);
    }
}
