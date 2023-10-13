using System;
using System.ComponentModel.DataAnnotations;
using Microsoft.VisualBasic;

namespace ValdeML
{   
    public class Model
    {
        public Model(Errors error)
        {
            ErrorType = (int)error;
            SelectedError = error.ToString();
        }
        public double Learning { get; set; }
        public int Epoch = 0;
        public int Epochs { get; set; }
        public int BatchId { get; set; }
        public int BatchSize { get; set; }
        public int ErrorType { get; set; }
        public string SelectedError { get; set; }
        public double Error { get; set; }
        public double[] Errors { get; set; }
        public double[][] ErrorDerivs { get; set; }
        public bool KeepTraining = true;
        #region Voids
        private void GetError(int activation)
        {
            if (activation.Equals(0))
            {
                Error = Errors.Sum() / (2 * Errors.Length);
            }
            else if (activation.Equals(1))
            {
                Error = Errors.Sum() / Errors.Length;
            }
        }
        public void UpdateError(double[][] layer_activations, double[] targets)
        {
            if(layer_activations.Length == targets.Length)
            {
                int size = layer_activations.Length;
                Errors = new double[size];
                ErrorDerivs = new double[size][];

                for (int i = 0; i < size; i++)
                {
                    double Error = 0.0;
                    double[] ErrorDeriv;
                    if (ErrorType.Equals(0))
                    {
                        Error = Math.Pow(layer_activations[i][0] - targets[i], 2);
                        ErrorDeriv = new double[1] { 2 * (layer_activations[i][0] - targets[i]) };
                        ErrorDerivs[i] = ErrorDeriv;
                    }
                    else if(ErrorType.Equals(1))
                    {
                        Error = targets[i].Equals(1) ? -Math.Log(layer_activations[i][0]) : -Math.Log(1 - layer_activations[i][0]);
                        ErrorDeriv = new double[1] { targets[i].Equals(1) ? -1 / layer_activations[i][0] : 1 / (1 - layer_activations[i][0]) } ;
                        ErrorDerivs[i] = ErrorDeriv;
                    }
                    Errors[i] = Error;
                    
                }
                GetError(ErrorType);
            }
            else
            {
                throw new Exception("Activations Length not equal to Targets Length.");
            }
        }
        public void SetLearningRate(double learning)
        {
            Learning = learning;
        }
        #endregion
    }
    public class Layer
    {
        public Layer(int layer_size, int layer_id, Activation layer_activation) 
        {
            LayerId = $"Layer [{layer_id}]";
            Nodes = new Node[layer_size];

            for (int i = 0; i < layer_size; i++)
            {
                Node node = new Node(i+1, 0, layer_activation);
                Nodes[i] = node;
            }
        }
        #region Layer Variables
        Node[] Nodes { get; set; }
        public double[][] Predictions { get; set; }
        public double[][] Activations { get; set; }
        public double[][] ActDerivs { get; set; }
        public double[][] NodeDerivs { get; set; }
        public double[][] NodeDerivsPow { get; set; }
        public double[][] NodeDerivsW { get; set; }
        public string LayerId { get; set; }
        #endregion
        public void Train(double[][] inputs)
        {
            int inputs_length = inputs.Length;
            int nodes_length = Nodes.Length;

            Predictions = new double[inputs_length][];
            Activations = new double[inputs_length][];
            ActDerivs = new double[inputs_length][];

            for (int i = 0; i < inputs_length; i++) 
            {
                double[] NodesPredictions = new double[nodes_length];
                double[] NodesActivations = new double[nodes_length];
                double[] NodesActDerivatives = new double[nodes_length];

                for (int j = 0; j < nodes_length; j++) 
                {
                    Node SelectedNode = Nodes[j];
                    double[] input = inputs[i];
                    SelectedNode.Predict(input);
                    
                    NodesPredictions[j] = SelectedNode.Pred;
                    NodesActivations[j] = SelectedNode.Act;
                    NodesActDerivatives[j] = SelectedNode.ActDer;
                }
                Predictions[i] = NodesPredictions;
                Activations[i] = NodesActivations;
                ActDerivs[i] = NodesActivations;
            }
        }
    
        public void GetNodeDerivatives(double[][] previus_derivative, double[][] respect_to)
        {
            var test = new Transposer().TransposeList(ActDerivs);
            int node_deriv_size = ActDerivs[0].Length;
            int node_derivs_size = ActDerivs.Length;

            NodeDerivs = new double[node_derivs_size][];
            NodeDerivsPow = new double[node_derivs_size][];

            for (int i = 0; i < node_derivs_size; i++)
            {
                double[] node_derivs = new double[node_deriv_size];
                double[] node_derivs_pow = new double[node_deriv_size];
                for (int j = 0; j < node_deriv_size; j++)
                {
                    double ActDeriv = ActDerivs[i][j];
                    double PrevDeriv = previus_derivative[i][j];

                    double NodeDeriv = PrevDeriv * ActDeriv;
                    double NodeDerivPow = Math.Pow(PrevDeriv * ActDeriv, 2);

                    node_derivs[j] = NodeDeriv;
                    node_derivs_pow[j] = NodeDerivPow;
                }
                NodeDerivs[i] = node_derivs;
                NodeDerivsPow[i] = node_derivs_pow;
            }
            
            double[][] inputsT = new Transposer().TransposeList(respect_to);
        }
    
        public void Update(Model model)
        {
            int nodes_size = Nodes.Length;
            for (int i = 0; i < nodes_size; i++)
            {
                Node SelectedNode = Nodes[i];

            }
        }
    }
    public class Node
    {
        public Node(int node_id, int features_size, Activation activation){

            NodeInfo = new NodeInfo(node_id, activation);
            ActType = (int)activation;
        }
        #region Gradient Variables
        public double[] W { get; set; }
        public double[] Vdw { get; set; }
        public double[] Sdw { get; set; }
        public double B { get; set; }
        public double[] Vdb { get; set; }
        public double[] Sdb { get; set; }
        #endregion
        #region Predictions and Errors Variables
        public double Pred { get; set; }
        public int ActType { get; set; }
        public double Act { get; set; }
        public double ActDer { get; set; }
        #endregion
        #region Control and Info Variables
        public NodeInfo NodeInfo { get; set; }
        internal int FeaturesSize { get; set; }
        private bool IsUpdated = false;
        public int FeatureId { get; set; }
        #endregion
        #region Voids
        private void UpdateWVars()
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
        public void Predict(double[] Input)
        {
            FeaturesSize = Input.Length;
            
            if (!IsUpdated)
                UpdateWVars();

            if(Input.Length == FeaturesSize)
            {
                double[] FeaturePreds = new double[FeaturesSize];
                for (int i = 0; i < FeaturesSize; i++)
                {
                    double FeaturePred = W[i] * Input[i];
                    FeaturePreds[i] = FeaturePred;
                }
                Pred = FeaturePreds.Sum() + B;
                if (ActType.Equals(3))
                {
                    Act = Math.Tanh(Pred);
                    ActDer = 1 - Math.Pow(Act, 2);
                }
                else if (ActType.Equals(4))
                {
                    Act = 1 / (1 + Math.Exp(-Pred));
                    ActDer = Act * (1 - Act);
                }
            }
            else
            {
                throw new Exception("Given Input size is not the same as Defined [FeaturesSize]");
            }
        }
        public void UpdateGradient(Model model, double[][] JWDers, double[][] JDers)
        {
            int w_size = W.Length;
            for (int i = 0; i < w_size; i++)
            {
                double tmp_w = W[i] - model.Learning * JWDers[i].Sum() / model.BatchSize;
                W[i] = tmp_w;
            }
            double tmp_b = B - model.Learning * JDers.Sum() / model.BatchSize;
            B = tmp_b;
        }
        #endregion
    }
    class Program
    {
        static void Main(string[] args)
        {
            Model model = new Model(Errors.LogLoss);
            model.SetLearningRate(.4);

            Layer layer1 = new Layer(4, 1, Activation.Tanh);
            Layer layer2 = new Layer(2, 2, Activation.Tanh);
            Layer layer3 = new Layer(1, 3, Activation.Sigmoid);

            DatasetMultFeatures data = new DatasetMultFeatures();
            data.Build(100000, 512, 2, "zscore", true);
            MMODEL[][] batches = data.batches;

            while (model.Error >= 0)
            {
                model.Epoch++;
                for (model.BatchId = 0; model.BatchId < batches.Length; model.BatchId++)
                {
                    MMODEL[] batch = batches[model.BatchId];
                    model.BatchSize = batch.Length;
                    double[][] inputs = batch.Select(x => x.input).ToArray();
                    double[] targets = batch.Select(x => x.target).ToArray();

                    layer1.Train(inputs);
                    layer2.Train(layer1.Activations);
                    layer3.Train(layer2.Activations);

                    model.UpdateError(layer3.Activations, targets);

                    layer3.GetNodeDerivatives(model.ErrorDerivs, layer2.Activations);
                    layer2.GetNodeDerivatives(layer3.NodeDerivs, layer1.Activations);
                }
                string msg = $"{model.Epoch}, {model.Error}";
                Console.WriteLine(msg);
            }
        }
    }
}
