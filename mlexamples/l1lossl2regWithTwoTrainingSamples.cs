using System;

class Program
{
    static void Main()
    {
        // Number of input neurons and output neurons
        int inputNeurons = 2;
        int outputNeurons = 2;  // 2 output neurons (e.g., for [0,1] classification)

        // Example training data (2 samples, each with 2 features)
        double[][] inputs = {
            new double[] { 0.5, 0.8 },  // First sample
            new double[] { 0.9, 0.1 }   // Second sample
        };

        // Target outputs (one-hot encoded, e.g., [0, 1] for class 1)
        double[][] targets = {
            new double[] { 0.0, 1.0 },  // First sample
            new double[] { 1.0, 0.0 }   // Second sample
        };

        // Random weights (2 weights per input neuron, for each output neuron)
        double[] weights = { 0.2, 0.4, 0.5, 0.6 };
        double[] bias = { 0.1, 0.2 };  // Bias for each output neuron

        // Hyperparameters
        double learningRate = 0.1;
        double regularizationLambda = 0.01;  // Regularization strength

        // Training loop (run for 1000 epochs)
        for (int epoch = 0; epoch < 1000; epoch++)
        {
            double totalLoss = 0.0;  // To accumulate the total loss for this epoch

            // Loop over each training sample
            for (int sampleIdx = 0; sampleIdx < inputs.Length; sampleIdx++)
            {
                double[] input = inputs[sampleIdx];
                double[] target = targets[sampleIdx];

                // **Forward Propagation**: Calculate weighted sum for each output neuron
                double[] outputs = new double[outputNeurons];
                double[] weightedSums = new double[outputNeurons];

                for (int i = 0; i < outputNeurons; i++)
                {
                    weightedSums[i] = 0.0;
                    for (int j = 0; j < inputNeurons; j++)
                    {
                        weightedSums[i] += input[j] * weights[i * inputNeurons + j];  // Adjust index for each output neuron
                    }
                    weightedSums[i] += bias[i];
                    outputs[i] = Sigmoid(weightedSums[i]);
                }

                // **Calculate Loss (L1 Loss + L2 Regularization)**
                double loss = 0.0;
                for (int i = 0; i < outputNeurons; i++)
                {
                    loss += L1Loss(outputs[i], target[i]);
                }

                // L2 Regularization (sum of squares of weights)
                double l2Regularization = 0.0;
                for (int i = 0; i < weights.Length; i++)
                {
                    l2Regularization += weights[i] * weights[i];
                }
                loss += regularizationLambda * l2Regularization;  // Add L2 penalty

                totalLoss += loss;  // Accumulate loss for this epoch

                // **Backpropagation**: Compute gradients and update weights
                double[] dLoss_dOutput = new double[outputNeurons];  // Gradients for output neurons

                for (int i = 0; i < outputNeurons; i++)
                {
                    // L1 Loss Gradient: sign(output - target)
                    dLoss_dOutput[i] = Math.Sign(outputs[i] - target[i]);

                    // Derivative of Sigmoid: sigmoid(output) * (1 - sigmoid(output))
                    double dOutput_dZ = SigmoidDerivative(outputs[i]);

                    // Compute gradients for weights and bias
                    for (int j = 0; j < inputNeurons; j++)
                    {
                        // Gradient of the loss w.r.t weight
                        double gradientW = dLoss_dOutput[i] * dOutput_dZ * input[j];
                        weights[i * inputNeurons + j] -= learningRate * gradientW;
                    }

                    // Gradient of the loss w.r.t bias
                    bias[i] -= learningRate * dLoss_dOutput[i] * SigmoidDerivative(outputs[i]);
                }
            }

            // **Output the loss and updated parameters after backpropagation (every 100 epochs)**
            if (epoch % 100 == 0)
            {
                Console.WriteLine($"Epoch {epoch + 1}");
                Console.WriteLine($"Total Loss (L1 Loss + L2 Regularization): {totalLoss:F4}");
                Console.WriteLine($"Updated Weights: W1 = {weights[0]:F4}, W2 = {weights[1]:F4}, W3 = {weights[2]:F4}, W4 = {weights[3]:F4}");
                Console.WriteLine($"Updated Bias: {bias[0]:F4}, {bias[1]:F4}");
                Console.WriteLine();
            }
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

    // L1 Loss function (for multi-output classification)
    static double L1Loss(double output, double target)
    {
        return Math.Abs(output - target); // |output - target|
    }
}
