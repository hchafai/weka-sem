package weka.filters.supervised.attribute;

/**
 * Weka-native discovery alias for the generic Composite SEM supervised filter.
 */
public class CompositeSEMFilter
        extends com.hopeproject.influencer.sem.weka.CompositeSEMFilter {

    private static final long serialVersionUID = 1L;

    public static void main(String[] args) {
        runFilter(new CompositeSEMFilter(), args);
    }
}
