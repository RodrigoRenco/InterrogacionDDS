using Classifier.ClassificationExceptions;
using System.Text.Json;

namespace Classifier.Encoders;

public class EncoderFactory
{
    public List<Sample> trainingDataset;
    public EncoderHandler EncoderHandler;

    public void CreateLabeledEncoderHandler(string label)
    {
        if (label == "label")
            EncoderHandler = new FitLabelEncoderHandler();
        else if(label == "one-hot")
            EncoderHandler = new FitOneHotEncoderHandler();
        else
        {
            throw new GenericException("Invalid encoder type");
        }
    }
}