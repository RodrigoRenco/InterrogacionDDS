namespace Classifier.ClassificationExceptions;

public class InvalidEncoderException : ClassificationExceptionBase
{
    private InvalidEncoderException():base("Invalid encoder type")
    {
    }
}