using System;
namespace ValdeML
{
    public class Node
    {
        #region Node Constructor
        public Node(int node_id, int features_size, Activation activation, Optimizer optimizer)
        {

            NodeInfo = new NodeInfo(node_id, activation);
            ActType = (int)activation;
            IsUpdated = true;
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
        #endregion
        public double Pred { get; set; }
        public int ActType { get; set; }
        public double Act { get; set; }
        public double ActDer { get; set; }
        public NodeInfo NodeInfo { get; set; }
        internal int FeaturesSize { get; set; }
        private bool IsUpdated { get; set; }
        public int FeatureId { get; set; }
        #endregion
        #region Node Voids
        private void UpdateWVars()
        { }
        public void Predict(double[] Input)
        { }
        public void UpdateGradient(Model model)
        { }
        #endregion
    }
}
