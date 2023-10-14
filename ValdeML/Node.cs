using System;
using System.Security.Cryptography.X509Certificates;

namespace ValdeML
{
    public class Node
    {
        #region Node Constructor
        public Node(int node_id, Activation activation, Optimizer optimizer)
        {

            NodeInfo = new NodeInfo(node_id, activation);
            IsUpdated = false;
            ActivationType = (int)activation;
            Optimizer = (int)optimizer;
            OptimizerName = optimizer.ToString();

            B = 0;
            Vdb = 0;
            Sdb = 0;
        }
        #endregion
        #region Node Variables

        #region Node Griadient Variables
        public double[] W { get; set; }
        public double[] Vdw { get; set; }
        public double[] Sdw { get; set; }
        public double B { get; set; }
        public double Vdb { get; set; }
        public double Sdb { get; set; }
        public int Optimizer { get; set; }
        public string OptimizerName { get; set; }
        #endregion
        #region Node Prediction and Activation Variables
        public int ActivationType { get; set; }
        public double[] Activations { get; set; }
        public double[] ActivationDerivs { get; set; }
        #endregion

        #region Node Control and Description Variables.
        public bool IsUpdated { get; set; }
        public int FeatureId { get; set; }
        public NodeInfo NodeInfo { get; set; }
        internal int FeaturesSize { get; set; }
        internal int TotalPredictions { get; set; }
        #endregion

        #region Node Derivatives Variables
        public double[] NodeJ          { get; set; }
        double[] NodeJPows      { get; set; }
        double[][] NodeJWs      { get; set; }
        double[][] NodeJWsPows  { get; set; }
        #endregion

        #region Node Voids
        public void Predict(double[] Input)
        { }
        public void UpdateGradient(Model model)
        {
            for (FeatureId = 0; FeatureId < W.Length; FeatureId++)
            {
                #region APPLY ADAM ON W
                double Jw = NodeJWs[FeatureId].Sum() / model.BatchSize;
                double JwPow = NodeJWsPows[FeatureId].Sum() / model.BatchSize;

                double old_vdw = model.B1 * Vdw[FeatureId] + (1 - model.B1) * Jw;
                double old_sdw = model.B2 * Sdw[FeatureId] + (1 - model.B2) * JwPow;
                
                Vdw[FeatureId] = old_vdw;
                Sdw[FeatureId] = old_sdw;

                double vdw_c = Vdw[FeatureId] / (1 - Math.Pow(model.B1, model.BatchSize));
                double sdw_c = Sdw[FeatureId] / (1 - Math.Pow(model.B2, model.BatchSize));
                #endregion
                //double tmp_w = W[FeatureId] - model.Learning * vdw_c / (Math.Sqrt(sdw_c) + model.e);
                double tw = NodeJWs[FeatureId].Sum() / model.BatchSize;
                double tmp_w = W[FeatureId] - model.Learning * tw;
                W[FeatureId] = tmp_w;
            }
            #region APPLY ADAM ON D
            double J = NodeJ.Sum() / model.BatchSize;
            double JPow = NodeJPows.Sum() / model.BatchSize;

            double old_vdb = model.B1 * Vdb + (1 - Math.Pow(model.B1, model.BatchSize)) * J;
            double old_sdb = model.B2 * Sdb + (1 - Math.Pow(model.B2, model.BatchSize)) * JPow;
            
            Vdb = old_vdb;
            Sdb = old_sdb;

            double vdb_c = Vdb / (1 - Math.Pow(model.B1, model.BatchSize));
            double sdb_c = Sdb / (1 - Math.Pow(model.B2, model.BatchSize));
            #endregion
            //double tmp_b = B - model.Learning * vdb_c / (Math.Sqrt(sdb_c) + model.e);
            double tj = NodeJ.Sum() / model.BatchSize;
            double tmp_b = B - model.Learning * tj;
            B = tmp_b;
        }
        #endregion

        #endregion
        #region Node Voids
        private void UpdateSlopes()
        {
            W = new double[FeaturesSize];
            Vdw = new double[FeaturesSize];
            Sdw = new double[FeaturesSize];

            Random random = new Random();

            for (int i = 0; i < FeaturesSize; i++)
            {
                W[i] = random.NextDouble() - 0.5;
                Vdw[i] = 0;
                Sdw[i] = 0;
            }

            IsUpdated = true;
        }
        private void ActivatePrediction(int activation_id, double prediction)
        {
            if (ActivationType.Equals(3))
            {
                Activations[activation_id] = Math.Tanh(prediction);
                ActivationDerivs[activation_id] = 1 - Math.Pow(Activations[activation_id], 2);
            }
            else if (ActivationType.Equals(4))
            {
                Activations[activation_id] = 1 / (1 + Math.Exp(-prediction));
                ActivationDerivs[activation_id] = Activations[activation_id] * (1 - Activations[activation_id]);
            }
        }
        public void NodePredict(double[] input, int feature_id, int total_inputs)
        {
            FeaturesSize = input.Length;

            if (!IsUpdated)
            {
                UpdateSlopes();
                Activations = new double[total_inputs];
                ActivationDerivs = new double[total_inputs];
            }

            double[] FeaturePredictions = new double[FeaturesSize];

            for (int i = 0; i < FeaturesSize; i++)
            {
                FeaturePredictions[i] = W[i] * input[i];
            }
            double Prediction = FeaturePredictions.Sum() + B;

            ActivatePrediction(feature_id, Prediction);
        }
        public void NodeCalDerivs(double[] previous_deriv, double[][] respect_to)
        {
            int NodejSize = ActivationDerivs.Length;
            
            NodeJ     = new double[NodejSize];
            NodeJPows = new double[NodejSize];

            for (int i = 0; i < NodejSize; i++)
            {
                NodeJ[i]     = previous_deriv[i] * ActivationDerivs[i];
                NodeJPows[i] = Math.Pow(previous_deriv[i] * ActivationDerivs[i], 2);
            }

            var InputsT      = new Transposer().TransposeList(respect_to);
            int SlopesLength = InputsT.Length;

            NodeJWs     = new double[SlopesLength][];
            NodeJWsPows = new double[SlopesLength][];

            for (int i = 0; i < SlopesLength; i++)
            {
                double[] Jws = new double[NodejSize];
                double[] JwsPows = new double[NodejSize];
                for (int j = 0; j < NodejSize; j++)
                {
                    Jws[j] = NodeJ[j] * InputsT[i][j];
                    JwsPows[j] = Math.Pow(NodeJ[j] * InputsT[i][j], 2);
                }
                NodeJWs[i]      = Jws;
                NodeJWsPows[i]  = JwsPows;
            }
        }
        #endregion
    }
}
