using System;

class Program
{
    static void Main()
    {
        // Number of neurons in the input and output layer
        int inputNeurons = 2;
        int outputNeurons = 1;

        // Random weights and biases for demonstration
        double[] input = { 0.5, 0.8 };  // Example input
        double[] weights = { 0.2, 0.4 }; // Random weights (1 per input neuron)
        double bias = 0.1;               // Bias for output node
        double learningRate = 0.1;       // Learning rate for gradient descent

        // Target value (for loss calculation)
        double target = 1.0;

        // Training loop (for demonstration, we run only one epoch)
        for (int epoch = 0; epoch < 1; epoch++)
        {
            // **Forward Propagation**: Calculate the weighted sum of inputs + bias
            double weightedSum = 0.0;
            for (int i = 0; i < inputNeurons; i++)
            {
                weightedSum += input[i] * weights[i];
            }
            weightedSum += bias;

            // Apply Sigmoid Activation function
            double output = Sigmoid(weightedSum);

            // **Calculate Loss (L1 Loss)**
            double loss = L1Loss(output, target);

            // **Backpropagation**:
            // 1. Compute the gradient of the L1 loss w.r.t output
            double dLoss_dOutput = Math.Sign(output - target);  // Derivative of L1 w.r.t output

            // 2. Compute the gradient of the Sigmoid function w.r.t output
            double dOutput_dZ = SigmoidDerivative(output); // Derivative of Sigmoid w.r.t z (pre-activation)

            // 3. Compute the gradient w.r.t weights and bias
            double dZ_dW1 = input[0];  // Derivative of Z w.r.t W1 (input[0] for weight[0])
            double dZ_dW2 = input[1];  // Derivative of Z w.r.t W2 (input[1] for weight[1])

            // Gradients for weights and bias
            double gradientW1 = dLoss_dOutput * dOutput_dZ * dZ_dW1;
            double gradientW2 = dLoss_dOutput * dOutput_dZ * dZ_dW2;
            double gradientBias = dLoss_dOutput * dOutput_dZ;

            // **Update Weights and Bias** using Gradient Descent
            weights[0] -= learningRate * gradientW1;
            weights[1] -= learningRate * gradientW2;
            bias -= learningRate * gradientBias;

            // **Output the loss and weights after backpropagation**
            Console.WriteLine($"Epoch {epoch + 1}");
            Console.WriteLine($"Output (after sigmoid): {output:F4}");
            Console.WriteLine($"Loss (L1 Loss): {loss:F4}");
            Console.WriteLine($"Updated Weights: W1 = {weights[0]:F4}, W2 = {weights[1]:F4}");
            Console.WriteLine($"Updated Bias: {bias:F4}");
            Console.WriteLine();
        }
    }

    // Sigmoid Activation function
    static double Sigmoid(double z)
    {
        return 1.0 / (1.0 + Math.Exp(-z));
    }

    // Derivative of Sigmoid function
    static double SigmoidDerivative(double output)
    {
        return output * (1 - output);
    }

    // L1 Loss function
    static double L1Loss(double output, double target)
    {
        return Math.Abs(output - target); // |output - target|
    }
}
