namespace Classifier.ClassificationExceptions;

public abstract class ClassificationExceptionBase(string message) : Exception (message)
{
    
    
    //LISTA excepciones:
    // 1- Al menos uno de los datasets no fue encontrado.
    // 2- Al menos un dataset est´a vac´ıo
    // 3- Se ingres´o un encoder inv´alido
    // 4- Se ingres´o un clasificador inv´alido
    // 5- Se ingres´o una m´etrica inv´alida
    // 6- Se escogi´o una regresi´on log´ıstica y no se especifico un learning rate
    // 7- Se escogi´o una regresi´on log´ıstica y el learning rate ingresado es negativo
    // 8- Se escogi´o una regresi´on log´ıstica y no se especifico la cantidad de ´epocas 
    // 9- Se escogi´o una regresi´on log´ıstica y la cantidad de ´epocas ingresada es negativa
    // 10- Se escogi´o KNN y no se especific´o la cantidad de vecinos
    // 11- Se escogi´o KNN y la cantidad de vecinos ingresada es negativa o nula
    // 12- Se escogi´o KNN y la cantidad de muestras del dataset de entrenamiento es menor a la cantidad de vecinos ingresada
}