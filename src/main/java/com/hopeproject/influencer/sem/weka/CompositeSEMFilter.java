package com.hopeproject.influencer.sem.weka;

import weka.core.Capabilities;
import weka.core.Option;
import weka.core.TechnicalInformation;
import weka.core.TechnicalInformationHandler;
import weka.core.Utils;
import weka.filters.SimpleBatchFilter;
import weka.filters.SupervisedFilter;
import weka.core.Instances;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Enumeration;
import java.util.List;

/**
 * Supervised Weka filter that appends configurable composite SEM scores.
 * The model is fitted only on the first batch and reused for subsequent batches.
 */
public class CompositeSEMFilter extends SimpleBatchFilter
        implements SupervisedFilter, TechnicalInformationHandler {

    private static final long serialVersionUID = 1L;

    private String constructs = "";
    private double ridge = 1.0e-6;
    private String prefix = "sem_";
    private boolean latentOnly;
    private boolean includePerformance = true;
    private CompositeSemModel model;

    @Override
    public String globalInfo() {
        return "Fits a configurable fixed-weight composite path model on the first batch and "
                + "adds standardized construct scores to Weka instances. This is a composite "
                + "path-modeling implementation, not a covariance-based SEM engine or canonical iterative PLS-PM.";
    }

    @Override
    protected Instances determineOutputFormat(Instances inputFormat) {
        CompositeSpecification specification = CompositeSpecification.parse(constructs);
        return CompositeWekaTransformer.outputFormat(
                inputFormat, specification, prefix, latentOnly, includePerformance);
    }

    @Override
    protected Instances process(Instances instances) {
        if (model == null) {
            model = new CompositeSemEstimator().fit(
                    instances, CompositeSpecification.parse(constructs), ridge);
        }
        return CompositeWekaTransformer.transform(instances, model, prefix, latentOnly, includePerformance);
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
        // Weka's command-line filter initializes from an empty header before
        // buffering the first batch. The estimator validates the real batch.
        capabilities.setMinimumNumberInstances(0);
        return capabilities;
    }

    @Override
    public Enumeration<Option> listOptions() {
        List<Option> options = new ArrayList<>();
        options.add(new Option(
                "Construct definitions: Name=attribute[:weight],attribute[:weight];Other=...",
                "constructs", 1, "-constructs <specification>"));
        options.add(new Option("Non-negative ridge added to the structural normal equations.",
                "ridge", 1, "-ridge <double>"));
        options.add(new Option("Prefix for generated attributes (default sem_).",
                "prefix", 1, "-prefix <text>"));
        options.add(new Option("Output latent variables only, plus the class.",
                "latent-only", 0, "-latent-only"));
        options.add(new Option("Do not add the structural performance score.",
                "no-performance", 0, "-no-performance"));
        Enumeration<Option> parent = super.listOptions();
        while (parent.hasMoreElements()) options.add(parent.nextElement());
        return Collections.enumeration(options);
    }

    @Override
    public void setOptions(String[] options) throws Exception {
        String value = Utils.getOption("constructs", options);
        if (!value.isBlank()) setConstructs(value);
        value = Utils.getOption("ridge", options);
        if (!value.isBlank()) setRidge(Double.parseDouble(value));
        value = Utils.getOption("prefix", options);
        if (!value.isBlank()) setPrefix(value);
        setLatentOnly(Utils.getFlag("latent-only", options));
        setIncludePerformance(!Utils.getFlag("no-performance", options));
        super.setOptions(options);
    }

    @Override
    public String[] getOptions() {
        List<String> options = new ArrayList<>();
        if (!constructs.isBlank()) {
            options.add("-constructs");
            options.add(constructs);
        }
        options.add("-ridge");
        options.add(Double.toString(ridge));
        options.add("-prefix");
        options.add(prefix);
        if (latentOnly) options.add("-latent-only");
        if (!includePerformance) options.add("-no-performance");
        Collections.addAll(options, super.getOptions());
        return options.toArray(String[]::new);
    }

    @Override
    public TechnicalInformation getTechnicalInformation() {
        TechnicalInformation information = new TechnicalInformation(TechnicalInformation.Type.ARTICLE);
        information.setValue(TechnicalInformation.Field.AUTHOR, "Henseler, J., Hubona, G., and Ray, P. A.");
        information.setValue(TechnicalInformation.Field.YEAR, "2016");
        information.setValue(TechnicalInformation.Field.TITLE,
                "Using PLS path modeling in new technology research: updated guidelines");
        information.setValue(TechnicalInformation.Field.JOURNAL, "Industrial Management & Data Systems");
        information.setValue(TechnicalInformation.Field.VOLUME, "116");
        information.setValue(TechnicalInformation.Field.NUMBER, "1");
        information.setValue(TechnicalInformation.Field.PAGES, "2-20");
        information.setValue(TechnicalInformation.Field.URL, "https://doi.org/10.1108/IMDS-09-2015-0382");
        return information;
    }

    public String constructsTipText() { return "Measurement-model specification for the composites."; }
    public String getConstructs() { return constructs; }
    public void setConstructs(String constructs) { this.constructs = constructs == null ? "" : constructs.trim(); }

    public String ridgeTipText() { return "Ridge used to stabilize the structural regression."; }
    public double getRidge() { return ridge; }
    public void setRidge(double ridge) {
        if (!Double.isFinite(ridge) || ridge < 0) throw new IllegalArgumentException("Ridge must be non-negative");
        this.ridge = ridge;
    }

    public String prefixTipText() { return "Prefix applied to generated latent-score attributes."; }
    public String getPrefix() { return prefix; }
    public void setPrefix(String prefix) { this.prefix = prefix == null ? "sem_" : prefix; }

    public String latentOnlyTipText() { return "Whether to remove original predictor attributes from the output."; }
    public boolean getLatentOnly() { return latentOnly; }
    public void setLatentOnly(boolean latentOnly) { this.latentOnly = latentOnly; }

    public String includePerformanceTipText() { return "Whether to append the structural performance score."; }
    public boolean getIncludePerformance() { return includePerformance; }
    public void setIncludePerformance(boolean includePerformance) { this.includePerformance = includePerformance; }

    public CompositeSemModel getModel() { return model; }

    @Override
    public String getRevision() { return "1.0"; }

    public static void main(String[] args) {
        runFilter(new CompositeSEMFilter(), args);
    }
}
