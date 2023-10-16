using System;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Security.Cryptography.X509Certificates;

namespace ValdeML
{
    public class Node
    {
        #region Node Constructor
        public Node(int node_id, Activation activation)
        {
            nodeId         = node_id;
            nodeIsReady    = false;
            activationId   = (int)activation;
            activationName = activation.ToString();
        }
        #endregion

        #region Node Info Variables
        public int nodeId                    { get; set; }
        public bool nodeIsReady              { get; set; }
        public int totalFeatures             { get; set; }
        #endregion

        #region Node Gradient Variables
        public double[] w                    { get; set; }
        public double[] vdw                  { get; set; }
        public double[] sdw                  { get; set; }
        public double b                      { get; set; }
        public double vdb                    { get; set; }
        public double sdb                    { get; set; }
        #endregion

        #region Node Activation Variables
        public int activationId              { get; set; }
        public string activationName         { get; set; }
        public double[] nodeActivations      { get; set; }
        public double[] nodeActivationDerivs { get; set; }
        #endregion

        #region Node Deltas Variables
        public double[] nodeDeltas           { get; set; }
        public double[][] weightDeltas       { get; set; }
        public double[] biasDeltas           { get; set; }
        #endregion

        #region Node Voids
        private void __UpdateNode()
        {
            Random random = new Random();
            w = new double[totalFeatures];
            vdw = new double[totalFeatures];
            sdw = new double[totalFeatures];

            for (int i = 0; i < totalFeatures; i++)
            {
                w[i]   = random.NextDouble() - 0.5;
                vdw[i] = 0;
                sdw[i] = 0;
            }

            b   = 0;
            vdb = 0;
            sdb = 0;

            nodeIsReady = true;
        }

        public void NodePredict(double[][] inputs)
        {
            totalFeatures             = inputs[0].Length;
            double[] activations      = new double[inputs.Length];
            double[] activationDerivs = new double[inputs.Length];

            if (!nodeIsReady)
                __UpdateNode();

            Span<double[]> inputsAsSpan = inputs;
            ref var searchspace         = ref MemoryMarshal.GetReference(inputsAsSpan);
            for (int i = 0; i < inputsAsSpan.Length; i++)
            {
                double[] features       = Unsafe.Add(ref searchspace, i);
                double[] featuresPred   = new double[totalFeatures];

                double activation       = 0.0;
                double activationDeriv  = 0.0;

                for (int j = 0; j < totalFeatures; j++)
                {
                    featuresPred[j] = w[j] * features[j];
                }
                double prediction = featuresPred.Sum() + b;

                if (activationId.Equals(3))
                {
                    activation      = Math.Tanh(prediction);
                    activationDeriv = 1 - Math.Pow(activation, 2);
                }
                else if (activationId.Equals(4))
                {
                    activation      = 1 / (1 + Math.Exp(-prediction));
                    activationDeriv = activation * (1 - activation);
                }

                activations[i]      = activation;
                activationDerivs[i] = activationDeriv;
            }
        }
        #endregion  
    }
}
