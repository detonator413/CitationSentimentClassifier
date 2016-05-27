/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package com.chaoticity.citationsentiment;

import weka.classifiers.Evaluation;
import weka.classifiers.functions.LibSVM;
import weka.classifiers.meta.FilteredClassifier;
import weka.core.Instances;
import weka.core.SerializationHelper;
import weka.core.converters.ConverterUtils;
import weka.core.tokenizers.NGramTokenizer;
import weka.core.tokenizers.WordTokenizer;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.StringToWordVector;

import java.io.ByteArrayOutputStream;
import java.io.ObjectOutput;
import java.io.ObjectOutputStream;
import java.util.Arrays;

import static com.chaoticity.citationsentiment.CitationSentimentTester.transformData;

/**
 * Code and data for citation sentiment classification reported in http://www.aclweb.org/anthology/P11-3015
 * The file test.arff contains only the test set with dependency triplets generated with Stanford CoreNLP
 * Full corpus available at http://www.cl.cam.ac.uk/~aa496/citation-sentiment-corpus
 *
 * @author Awais Athar
 */
public class CitationSentimentTrainer {


    /**
     * @param args the command line arguments
     */
    public static void main(String[] args) throws Exception {
        ConverterUtils.DataSource source = new ConverterUtils.DataSource("test.arff");
        Instances data = source.getDataSet();

        // Set class attribute
        data.setClassIndex(data.attribute("@@class@@").index());

        // delete unused attributes
        data.deleteAttributeAt(1);
        data.deleteAttributeAt(2);

        // split dependencies on space
        StringToWordVector unigramFilter = new StringToWordVector();
        unigramFilter.setInputFormat(data);
        unigramFilter.setIDFTransform(true);
        unigramFilter.setAttributeIndices("3");
        WordTokenizer whitespaceTokenizer = new WordTokenizer();
        whitespaceTokenizer.setDelimiters(" ");
        unigramFilter.setTokenizer(whitespaceTokenizer);
        data = Filter.useFilter(data,unigramFilter);

        // make trigrams from citation sentences
        StringToWordVector trigramFilter = new StringToWordVector();
        trigramFilter.setInputFormat(data);
        trigramFilter.setIDFTransform(true);
        trigramFilter.setAttributeIndices("2");
        NGramTokenizer tokenizer = new NGramTokenizer();
        tokenizer.setNGramMinSize(1);
        tokenizer.setNGramMaxSize(3);
        trigramFilter.setTokenizer(tokenizer);
        data = Filter.useFilter(data,trigramFilter);


        LibSVM svm = new LibSVM();
        svm.setCost(1000);

        FilteredClassifier classifier  = new FilteredClassifier();
        classifier.setClassifier(svm);
        classifier.buildClassifier(data);
        SerializationHelper.write("/tmp/citmodel.dat", classifier);


        ConverterUtils.DataSource testDataSource = new ConverterUtils.DataSource("example.arff");
        Instances testData = transformData(testDataSource.getDataSet());

        for (int i = 0; i < testData.numInstances(); i++) {
            double[] res = classifier.distributionForInstance(testData.instance(i));
            System.out.println(Arrays.toString(res));
        }

    }


}
