using System;
namespace ValdeML
{
    #region Common Models
    public class NodeInfo
    {
        public NodeInfo(int node_id, Activation activation)
        {
            NodeId = node_id;
            ActivationType = activation.ToString();
            int act_id = (int)activation;
            if (act_id == 0)
            {
                ActivationMath = "-";
                ActivationDerivMath = "-";
            }
            else if (act_id == 3)
            {
                ActivationMath = "Math.Tanh(prediction);";
                ActivationDerivMath = " 1 - Math.Pow(activation, 2);";
            }
            else if (act_id == 4)
            {
                ActivationMath = "1 / (1 + Math.Exp(-prediction));";
                ActivationDerivMath = "activation * (1 - activation);";
            }
        }
        internal int NodeId { get; set; }
        internal string ActivationType { get; set; }
        internal string ActivationMath { get; set; }
        internal string ActivationDerivMath { get; set; }
    }
    public class LayerInfo
    {
        public LayerInfo(Activation activation)
        {
            Activation = activation.ToString();
            int SelectedActivationId = (int)activation;

            if(SelectedActivationId.Equals(3))
            {
                NodeActivationCalc = "Math.Tanh(Prediction)";
                NodeActivationDerivCalc = "1 - Math.Pow(Activation,2)";
            }
            else if(SelectedActivationId.Equals(4))
            {
                NodeActivationCalc = "1 / (1 + Math.Exp(-Prediction))";
                NodeActivationDerivCalc = "Activation * (1 - Activation)";
            }
        }

        internal string Activation { get; set; }
        internal string NodeActivationCalc { get; set; }
        internal string NodeActivationDerivCalc { get; set; }
    }
    public class SCALER
    {
        internal string type { get; set; }
        internal double m { get; set; }
        internal double s { get; set; }
        internal double min { get; set; }
        internal double max { get; set; }
    }
    public class SMODEL
    {
        internal double input { get; set; }
        internal double target { get; set; }
    }
    public class MMODEL
    {
        internal double[] input;
        internal double target;
    }
    #endregion
    public class Model
    {
        #region Model Constructor
        public Model(Errors error)
        {
            ErrorType = (int)error;
            SelectedError = error.ToString();
            KeepTraining = true;
        }
        #endregion
        #region Model Variables
        public int Epoch = 0;
        public int Epochs { get; set; }

        public int BatchId { get; set; }
        public int BatchSize { get; set; }

        public int ErrorType { get; set; }
        public string SelectedError { get; set; }
        public double Error { get; set; }

        public double[] Errors { get; set; }
        public double[][] ErrorDerivs { get; set; }

        public double Learning { get; set; }
        public bool KeepTraining { get; set; }

        public double B1 = 0.9;
        public double B2 = 0.999;
        public double e = Math.Pow(10, -8);
        #endregion
        
    }
}

