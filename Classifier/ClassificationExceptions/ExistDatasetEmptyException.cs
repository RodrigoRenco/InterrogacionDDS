namespace Classifier.ClassificationExceptions;

public class ExistDatasetEmptyException : ClassificationExceptionBase
{
    public ExistDatasetEmptyException(string message):base(message)
    {
        message = $"Dataset file not found: {message}-train";
    }
}