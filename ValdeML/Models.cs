using System;
namespace ValdeML
{
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
        public string LayerID { get; set; }
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
}

