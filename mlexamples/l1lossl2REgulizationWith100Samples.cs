using System;

class Program
{
    static void Main()
    {
        // Number of input neurons and output neurons
        int inputNeurons = 2;
        int outputNeurons = 2;  // 2 output neurons (e.g., for [0,1] classification)

        // Example training data (100 samples, each with 2 features)
        Random rand = new Random();
        double[][] inputs = new double[100][];
        double[][] targets = new double[100][];
        for (int i = 0; i < 100; i++)
        {
            inputs[i] = new double[] { rand.NextDouble(), rand.NextDouble() };  // Random values between 0 and 1
            targets[i] = new double[] { rand.Next(2), rand.Next(2) };           // Random [0,1] or [1,0] labels
        }

        // Random weights (2 weights per input neuron, for each output neuron)
        double[] weights = { 0.2, 0.4, 0.5, 0.6 };
        double[] bias = { 0.1, 0.2 };  // Bias for each output neuron

        // Hyperparameters
        double learningRate = 0.1;
        double regularizationLambda = 0.01;  // Regularization strength
        double initialLearningRate = 0.1;
        double learningRateDecay = 0.995;    // Learning rate decay
        int maxEpochs = 1000;                // Max epochs
        int patience = 50;                   // Early stopping patience
        double bestValidationLoss = double.MaxValue;
        int wait = 0;

        // Training loop (run for 1000 epochs)
        for (int epoch = 0; epoch < maxEpochs; epoch++)
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
            }

            // **Average Loss**: Normalize by number of samples
            double averageLoss = totalLoss / inputs.Length;
            Console.WriteLine($"Epoch {epoch + 1}");
            Console.WriteLine($"Average Loss (L1 Loss + L2 Regularization): {averageLoss:F4}");

            // Implement Early Stopping based on validation loss
            double validationLoss = CalculateValidationLoss(inputs, targets); // Using same data for simplicity, ideally should be separate validation set
            if (validationLoss < bestValidationLoss)
            {
                bestValidationLoss = validationLoss;
                wait = 0;
            }
            else
            {
                wait++;
                if (wait >= patience)
                {
                    Console.WriteLine("Early stopping due to no improvement in validation loss.");
                    break;
                }
            }

            // Implement Learning Rate Decay
            learningRate *= learningRateDecay;
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

    // Simple function to calculate the loss for validation set (in this case using training data itself)
    static double CalculateValidationLoss(double[][] inputs, double[][] targets)
    {
        double totalLoss = 0;
        for (int i = 0; i < inputs.Length; i++)
        {
            double[] output = new double[2];  // For 2 output neurons
            double[] weightedSums = new double[2];
            for (int j = 0; j < 2; j++) 
            {
                weightedSums[j] = 0.0;
                for (int k = 0; k < 2; k++)
                {
                    weightedSums[j] += inputs[i][k] * 0.2;  // Just a dummy weight for validation
                }
                weightedSums[j] += 0.1;  // Dummy bias
                output[j] = Sigmoid(weightedSums[j]);
            }
            for (int j = 0; j < 2; j++)
            {
                totalLoss += L1Loss(output[j], targets[i][j]);
            }
        }
        return totalLoss;
    }
}
