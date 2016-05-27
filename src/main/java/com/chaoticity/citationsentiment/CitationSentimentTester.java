/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package com.chaoticity.citationsentiment;

import weka.classifiers.functions.LibSVM;
import weka.classifiers.meta.FilteredClassifier;
import weka.core.Instances;
import weka.core.SerializationHelper;
import weka.core.converters.ConverterUtils;
import weka.core.tokenizers.NGramTokenizer;
import weka.core.tokenizers.WordTokenizer;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.StringToWordVector;

import java.util.Arrays;

/**
 * Code and data for citation sentiment classification reported in http://www.aclweb.org/anthology/P11-3015
 * The file test.arff contains only the test set with dependency triplets generated with Stanford CoreNLP
 * Full corpus available at http://www.cl.cam.ac.uk/~aa496/citation-sentiment-corpus
 *
 * @author Awais Athar
 */
public class CitationSentimentTester {


    /**
     * @param args the command line arguments
     */
    public static void main(String[] args) throws Exception {
        FilteredClassifier svm = (FilteredClassifier) SerializationHelper.read("/tmp/citmodel.dat");
        ConverterUtils.DataSource source = new ConverterUtils.DataSource("example.arff");
        Instances data = transformData(source.getDataSet());

        for (int i = 0; i < data.numInstances(); i++) {
            double[] res = svm.distributionForInstance(data.instance(i));
            System.out.println(Arrays.toString(res));
        }


    }

    public static Instances transformData(Instances data) throws Exception {
        data.deleteAttributeAt(0);

        // split dependencies on space
        StringToWordVector unigramFilter = new StringToWordVector();
        unigramFilter.setInputFormat(data);
        unigramFilter.setIDFTransform(true);
        unigramFilter.setAttributeIndices("2");
        WordTokenizer whitespaceTokenizer = new WordTokenizer();
        whitespaceTokenizer.setDelimiters(" ");
        unigramFilter.setTokenizer(whitespaceTokenizer);
        data = Filter.useFilter(data,unigramFilter);

        // make trigrams from citation sentences
        StringToWordVector trigramFilter = new StringToWordVector();
        trigramFilter.setInputFormat(data);
        trigramFilter.setIDFTransform(true);
        trigramFilter.setAttributeIndices("1");
        NGramTokenizer tokenizer = new NGramTokenizer();
        tokenizer.setNGramMinSize(1);
        tokenizer.setNGramMaxSize(3);
        trigramFilter.setTokenizer(tokenizer);
        data = Filter.useFilter(data,trigramFilter);
        return data;
    }


}
