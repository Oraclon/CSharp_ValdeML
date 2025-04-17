using System;

class Program
{
    // Method to calculate L1 Loss (Absolute Error)
    public static double CalculateL1Loss(double predicted, double actual)
    {
        return Math.Abs(predicted - actual);  // Absolute difference
    }

    // Method to calculate the derivative (gradient) of L1 Loss
    public static double DerivativeL1Loss(double predicted, double actual)
    {
        if (predicted < actual)
        {
            return 1;  // Positive error, increase prediction
        }
        else if (predicted > actual)
        {
            return -1; // Negative error, decrease prediction
        }
        else
        {
            return 0;  // No error (exact match)
        }
    }

    static void Main(string[] args)
    {
        // Example predicted and actual values
        double predicted = 4.5;  // Example predicted value
        double actual = 5.0;     // Example actual value (ground truth)

        // Calculate the L1 Loss
        double l1Loss = CalculateL1Loss(predicted, actual);
        Console.WriteLine($"L1 Loss: {l1Loss}");

        // Calculate the derivative (gradient) of the L1 Loss
        double gradient = DerivativeL1Loss(predicted, actual);
        Console.WriteLine($"Derivative of L1 Loss (Gradient): {gradient}");

        // Example when predicted equals actual
        double predictedExact = 5.0;  // Example predicted value equal to actual
        double gradientExact = DerivativeL1Loss(predictedExact, actual);
        Console.WriteLine($"Derivative of L1 Loss when predicted equals actual: {gradientExact}");
    }
}
