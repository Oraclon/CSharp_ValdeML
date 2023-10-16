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

        #region Prediction Calculations
        private void __PrepareNode()
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
                __PrepareNode();

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

            nodeActivations         = activations;
            nodeActivationDerivs    = activationDerivs;
        }
        #endregion

        #region Delta Calculations
        private void _GetNodeDeltas(double[][] tmp_delta_calculation_results)
        {
            double[][] tmp_delta_calculation_results_T = Transposer.TransposeList(tmp_delta_calculation_results);

            int outerSize = tmp_delta_calculation_results_T.Length;
            int innerSize = tmp_delta_calculation_results_T[0].Length;

            double[] deltas = new double[outerSize];
            for (int i = 0; i < outerSize; i++)
            {
                deltas[i] = tmp_delta_calculation_results_T[i].Sum() / innerSize;
            }
            nodeDeltas = deltas;
        }

        public void NodeCalcDeltas(double[][] previous_derivatives, double[][] respect_to)
        {
            int outer_prev_derivs_size = previous_derivatives.Length;
            int inner_prev_derivs_size = previous_derivatives[0].Length;

            double[][] tmp_delta_calculations = new double[outer_prev_derivs_size][];

            for (int i = 0; i < outer_prev_derivs_size; i++)
            {
                double[] delta_calculations = new double[inner_prev_derivs_size];

                Span<double> prevDersAsSpan = previous_derivatives[i];
                ref var prevDersSearchArea = ref MemoryMarshal.GetReference(prevDersAsSpan);
                for (int j = 0; j < prevDersAsSpan.Length; j++)
                {
                    double prevDerivative = Unsafe.Add(ref prevDersSearchArea, j);
                    delta_calculations[j] = prevDerivative * nodeActivationDerivs[j];
                }

                tmp_delta_calculations[i] = delta_calculations;
            }
            //nodeDeltas = _GetNodeDeltas(tmp_delta_calculations);
            _GetNodeDeltas(tmp_delta_calculations);

            double[][] inputsT = Transposer.TransposeList(respect_to);

            int inputsTOuter = inputsT.Length;
            int inputsTInner = inputsT[0].Length;

            double[][] tmp_weight_deltas = new double[inputsTOuter][];

            for (int i = 0; i < inputsTOuter; i++) 
            {
                double[] weight_delta = new double[inputsTInner];
                for (int j = 0; j < inputsTInner; j++)
                {
                    weight_delta[j] = nodeDeltas[j] * inputsT[i][j];
                }
                tmp_weight_deltas[i] = weight_delta;
            }
            weightDeltas = tmp_weight_deltas;
        }
        #endregion

        public void NodeUpdate(Model model)
        {
            for (int i = 0; i < totalFeatures; i++)
            {
                double jw = weightDeltas[i].Sum() / model.BatchSize;
                double tmp_w = w[i] - model.Learning * jw;
                w[i] = tmp_w;
            }

            double j = nodeDeltas.Sum() / model.BatchSize;
            double tmp_b = b - model.Learning * j;
            b = tmp_b;
        }
        #endregion  
    }
}
