using System.Text.Json;
namespace Classifier.ClassificationExceptions;

public class ExistInvalidClasiffierException : ClassificationExceptionBase
{
    public ExistInvalidClasiffierException( ):base("Invalid classifier type")
    {
    }
}