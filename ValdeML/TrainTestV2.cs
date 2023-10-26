//NodeCore.cs
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace AimsMLV2
{

    #region Node Classes
    public static class OPTV
    {
        public static double B1 = 0.9;
        public static double B2 = 0.999;
        public static double E = Math.Pow(10, -8);
    }
    public class Inf
    {
        public Errors err        { get; set; }
        public Activation act    { get; set; }
        public Optimizer opt     { get; set; }
    }
    public class LoopData
    {
        public double[][] inputs { get; set; }
        public double[] targets  { get; set; }
    }
    public class Sizes
    {
        public int featuresLen { get; set; }
        public int dataLen { get; set; }
    }
    public class Gradient
    {
        public double[] w { get; set; }
        public double[] vdw { get; set; }
        public double[] sdw { get; set; }
        public double b { get; set; }
        public double vdb { get; set; }
        public double sdb { get; set; }

        public void GenerateGrad(Sizes sizes)
        {
            Random rand = new Random();
            w = new double[sizes.featuresLen];
            vdw = new double[sizes.featuresLen];
            sdw = new double[sizes.featuresLen];

            for (int i = 0; i < sizes.featuresLen; i++)
            {
                w[i] = rand.NextDouble() - 0.5;
                vdw[i] = 0;
                sdw[i] = 0;
            }

            b = 0;
            vdb = 0;
            sdb = 0;
        }
    }
    public class Storage
    {
        public double[] activations { get; set; }
        public double[] activationDerivs { get; set; }
        public double[] errors { get; set; }
        public double[] errorDerivs { get; set; }
        public double[] deltas { get; set; }
        public double[] deltasPow { get; set; }
        public double[][] wDeltas { get; set; }
        public double[][] wDeltasPow { get; set; }
    }
    #endregion

    public class Node
    {
        #region Node Constructor
        public Node(Activation activation, Optimizer optimizer)
        {
            isReady = false;
            keepTrain = true;
            inf = new Inf();
            sizes = new Sizes();
            grad = new Gradient();
            storage = new Storage();
            Learning = .4;
            inf.act = activation;
            inf.opt = optimizer;
        }
        #endregion

        #region Node Variables
        public double Learning { get; set; }
        public Gradient grad { get; set; }
        public Sizes sizes { get; set; }
        public int batchId { get; set; }
        public int itemId { get; set; }
        public int featureId { get; set; }
        public bool isReady { get; set; }
        public bool keepTrain { get; set; }
        public Storage storage { get; set; }
        public Inf inf { get; set; }
        public LoopData GetLoopData(Dataset dataSet)
        {
            Data[] data = dataSet.vars.hasBatches ? dataSet.vars.batches[batchId] : dataSet.vars.dataset;
            sizes.dataLen = data.Length;
            sizes.featuresLen = data[0].input.Length;

            if (!isReady)
            {
                grad.GenerateGrad(sizes);
                isReady = true;
            }

            LoopData loopD = new LoopData();
            loopD.inputs = data.Select(x => x.input).ToArray();
            loopD.targets = data.Select(x => x.target).ToArray();
            return loopD;
        }

        #endregion

        #region Node Common Voids
        public double GetJW(bool isPow = false)
        {
            int deb = featureId;

            if (!isPow)
                return storage.wDeltas[featureId].Sum() / sizes.dataLen;
            else
                return storage.wDeltasPow[featureId].Sum() / sizes.dataLen;
        }
        public double GetJ(bool isPow = false)
        {
            if (!isPow)
                return storage.deltas.Sum() / sizes.dataLen;
            else
                return storage.deltasPow.Sum() / sizes.dataLen;
        }
        #endregion

        #region Node Optimizers
        public void Default(Model model)
        {
            for (featureId = 0; featureId < sizes.featuresLen; featureId++)
            {
                double tmp_w = grad.w[featureId] - model.Learning * GetJW();
                grad.w[featureId] = tmp_w;
            }
            double tmp_b = grad.b - model.Learning * GetJ();
            grad.b = tmp_b;
        }
        public void Momentum(Model model)
        {
            for (featureId = 0; featureId < sizes.featuresLen; featureId++)
            {
                double old_vdw = OPTV.B1 * grad.vdw[featureId] + (1 - OPTV.B1) * GetJW();
                grad.vdw[featureId] = old_vdw;
                double tmp_w = grad.w[featureId] - model.Learning * grad.vdw[featureId];
                grad.w[featureId] = tmp_w;
            }
            double old_vdb = OPTV.B1 * grad.vdb + (1 - OPTV.B1) * GetJ();
            grad.vdb = old_vdb;
            double tmp_b = grad.b - model.Learning * grad.vdb;
            grad.b = tmp_b;
        }
        public void RmsProp(Model model)
        {
            for (featureId = 0; featureId < sizes.featuresLen; featureId++)
            {
                double old_sdw = OPTV.B2 * grad.sdw[featureId] + (1 - OPTV.B2) * GetJW(true);
                grad.sdw[featureId] = old_sdw;
                double tmp_w = grad.w[featureId] - model.Learning * GetJW() / Math.Sqrt(grad.sdw[featureId]);
                grad.w[featureId] = tmp_w;
            }

            double old_sdb = OPTV.B2 * grad.sdb + (1 - OPTV.B2) * GetJ(true);
            grad.sdb = old_sdb;
            double tmp_b = grad.b - model.Learning * GetJ() / Math.Sqrt(grad.sdb);
        }
        public void Adam(Model model)
        {
            for (featureId = 0; featureId < sizes.featuresLen; featureId++)
            {
                double old_vdw = OPTV.B1 * grad.vdw[featureId] + (1 - OPTV.B1) * GetJW();
                double old_sdw = OPTV.B2 * grad.sdw[featureId] + (1 - OPTV.B2) * GetJW(true);
                grad.vdw[featureId] = old_vdw;
                grad.sdw[featureId] = old_sdw;
                double vdw_c = grad.vdw[featureId] / (1 - Math.Pow(OPTV.B1, sizes.dataLen));
                double sdw_c = grad.sdw[featureId] / (1 - Math.Pow(OPTV.B2, sizes.dataLen));
                double tmp_w = grad.w[featureId] - model.Learning * vdw_c / (Math.Sqrt(sdw_c) + OPTV.E);
                grad.w[featureId] = tmp_w;
            }

            double old_vdb = OPTV.B1 * grad.vdb + (1 - OPTV.B1) * GetJ();
            double old_sdb = OPTV.B2 * grad.sdb + (1 - OPTV.B2) * GetJ(true);
            grad.vdb = old_vdb;
            grad.sdb = old_sdb;
            double vdb_c = grad.vdb / (1 - Math.Pow(OPTV.B1, sizes.dataLen));
            double sdb_c = grad.sdb / (1 - Math.Pow(OPTV.B2, sizes.dataLen));
            double tmp_b = grad.b - model.Learning * vdb_c / (Math.Sqrt(sdb_c) + OPTV.E);
            grad.b = tmp_b;
        }
        #endregion

        #region Node Gradient Update
        public void UpdateGradient(Model model)
        {
            switch(inf.opt)
            {
                case Optimizer.Default:
                    Default(model);
                    break;
                case Optimizer.Momentum:
                    Momentum(model);
                    break;
                case Optimizer.RmsProp:
                    RmsProp(model);
                    break;
                case Optimizer.Adam:
                    Adam(model);
                    break;
            }
        }
        #endregion

        #region Node Deltas
        private void _AverageSum(double[][] lDeltas)
        {
            double[][] deltasT = DataActions.TransposeList(lDeltas);
            for (itemId = 0; itemId < sizes.dataLen; itemId++)
            {
                double[] deltaT = deltasT[itemId];
                double delta = deltaT.Length > 1 ? deltaT.Sum() / deltaT.Length : deltaT[0];
                storage.deltas[itemId] = delta;
                storage.deltasPow[itemId] = Math.Pow(delta,2);
            }
        }
        private void _GetWDeltas(double[][] respectTo)
        {
            double[][] inputsT = DataActions.TransposeList(respectTo);
            for (featureId = 0; featureId < sizes.featuresLen; featureId++)
            {
                double[] inputT = inputsT[featureId];
                double[] tmpW = new double[sizes.dataLen];
                double[] tmpWP = new double[sizes.dataLen];

                for (itemId = 0; itemId < sizes.dataLen; itemId++)
                {
                    double delta = storage.deltas[itemId];
                    double input = inputT[itemId];
                    tmpW[itemId] = delta * input;
                    tmpWP[itemId] = Math.Pow(delta * input, 2);
                }
                storage.wDeltas[featureId] = tmpW;
                storage.wDeltasPow[featureId] = tmpWP;
            }
        }
        public void LDeltas(double[][] prevDerivs, double[][] respectTo)
        {
            int oSize = prevDerivs.Length;
            double[][] tmpDeltas = new double[oSize][];
            for(int i = 0; i < oSize; i++)
            {
                double[] tmpDelta = new double[sizes.dataLen];
                for(itemId= 0; itemId < sizes.dataLen; itemId++)
                {
                    double prevDeriv = prevDerivs[i][itemId];
                    double actDeriv = storage.activationDerivs[itemId];
                    tmpDelta[itemId] = prevDeriv * actDeriv;
                }
                tmpDeltas[i] = tmpDelta;
            }
            _AverageSum(tmpDeltas);
            _GetWDeltas(respectTo);
        }
        public void Deltas(double[] prevDerivs, double[][] respectTo)
        {
            for (itemId = 0; itemId < sizes.dataLen; itemId++)
            {
                double prevDeriv = prevDerivs[itemId];
                double actDeriv  = storage.activationDerivs[itemId];
                storage.deltas[itemId] = prevDeriv * actDeriv;
                storage.deltasPow[itemId] = Math.Pow(prevDeriv * actDeriv, 2);
            }

            _GetWDeltas(respectTo);
        }
        #endregion

        #region Node Training
        private void _BuildGradient(Model model, double[][] inputs)
        {
            sizes.dataLen = inputs.Length;
            sizes.featuresLen = inputs[0].Length;

            grad.w = new double[sizes.featuresLen];
            grad.vdw = new double[sizes.featuresLen];
            grad.sdw = new double[sizes.featuresLen];

            storage.activations = new double[sizes.dataLen];
            storage.activationDerivs = new double[sizes.dataLen];

            storage.deltas = new double[sizes.dataLen];
            storage.deltasPow = new double[sizes.dataLen];

            storage.wDeltas = new double[sizes.featuresLen][];
            storage.wDeltasPow = new double[sizes.featuresLen][];

            for (int i = 0; i < sizes.featuresLen; i++)
            {
                grad.w[i] = model.rand.NextDouble() - .5;
                grad.vdw[i] = 0;
                grad.sdw[i] = 0;
            }

            grad.b = 0;
            grad.vdb = 0;
            grad.sdb = 0;

            isReady = true;
        }
        private void _ActivatePrediction(double prediction)
        {
            double activation = 0;
            double activationDeriv = 0;
            switch(inf.act)
            {
                case Activation.Default:
                    activation = prediction;
                    activationDeriv = 1;
                    break;
                case Activation.Tanh:
                    activation = Math.Tanh(prediction);
                    activationDeriv = 1 - Math.Pow(activation, 2);
                    break;
                case Activation.Sigmoid:
                    activation = 1 / (1 + Math.Exp(-prediction));
                    activationDeriv = activation * (1 - activation);
                    break;
            }
            storage.activations[itemId] = activation;
            storage.activationDerivs[itemId] = activationDeriv;
        }
        public void Train(Model model, double[][] inputs)
        {
            if (!isReady)
                _BuildGradient(model, inputs);

            for (itemId = 0; itemId < sizes.dataLen; itemId++)
            {
                double[] featuresCalc = new double[sizes.featuresLen];
                for (featureId = 0; featureId < sizes.featuresLen; featureId++)
                {
                    featuresCalc[featureId] = grad.w[featureId] * inputs[itemId][featureId];
                }
                double prediction = featuresCalc.Sum() + grad.b;
                _ActivatePrediction(prediction);
            }
        }
        #endregion
    }
}

//TrainerCore.cs
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace AimsMLV2
{
    public class Trainer
    {
        #region Trainer Constructor
        public Trainer()
        {
            LayerModels = new List<LayerModel>();
        }
        #endregion

        #region Trainer Variables
        public List<LayerModel> LayerModels { get; set; }
        public Layer[] Layers { get; set; }
        public Model model { get; set; }
        int totalNodes { get; set; }
        int layerId { get; set; }
        #endregion

        #region Trainer Voids
        public void AddLayer(int nodes, Activation activation, Optimizer optimizer)
        {
            LayerModel layerModel = new LayerModel();
            layerModel.nodes = nodes;
            layerModel.activation = activation;
            layerModel.optimizer = optimizer;
            LayerModels.Add(layerModel);
        }

        public void BuildLayers(Model md)
        {
            totalNodes = LayerModels.Count();
            Layers = new Layer[totalNodes];
            model = md;

            for (int i = 0; i < totalNodes; i++)
            {
                LayerModel layerModel = LayerModels[i];
                Layers[i] = new Layer(md, layerModel.nodes, layerModel.activation, layerModel.optimizer);
            }
        }

        public void StartTrain(Dataset dataset)
        {
            do
            {
                model.Epoch++;
                for (model.BatchId = 0; model.BatchId < dataset.vars.batches.Length; model.BatchId++)
                {
                    if (model.keepTrain)
                    {
                        Data[] batch = dataset.vars.batches[model.BatchId];
                        double[][] inputs = batch.Select(x => x.input).ToArray();
                        double[] targets = batch.Select(x => x.target).ToArray();
                        LayerPredict(inputs, targets);
                        LayerDeltas(inputs);
                        LayerUpdate();
                    }
                    else
                        break;
                }
                Console.Write($"\r Cost: {model.Cost}");
            }
            while (model.Epochs.Equals(0) ? model.Cost >= 0 && model.keepTrain : model.Epoch < model.Epochs);
        }

        public void LayerPredict(double[][] inputs, double[] targets)
        {
            for (layerId = 0; layerId < totalNodes; layerId++)
            {
                Layer selectedLayer = Layers[layerId];
                if (layerId.Equals(0))
                    selectedLayer.LayerPredict(inputs);
                else
                    selectedLayer.LayerPredict(Layers[layerId - 1].storage.nodeActivations);
            }

            model.GetLayerError(Layers[totalNodes - 1].storage.nodeActivations, targets);
        }

        public void LayerDeltas(double[][] inputs)
        {
            int total = totalNodes - 1;
            for (layerId = total; layerId >= 0; layerId--)
            {
                if (layerId.Equals(total))
                    Layers[layerId].LayerDeltas(model.storage.lErrorDerivs, Layers[layerId - 1].storage.nodeActivations);
                else if (layerId.Equals(0))
                    Layers[layerId].LayerDeltas(Layers[layerId + 1].storage.nodeDeltas, inputs);
                else
                    Layers[layerId].LayerDeltas(Layers[layerId + 1].storage.nodeDeltas, Layers[layerId - 1].storage.nodeActivations);
            }
        }

        public void LayerUpdate()
        {
            int total = totalNodes - 1;
            for (layerId = total; layerId >= 0; layerId--)
            {
                Layers[layerId].UpdateNodes();
            }
        }
        #endregion
    }
}

//LayerCore.cs
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace AimsMLV2
{
    public class LayerModel
    {
        public int nodes { get; set; }
        public Activation activation { get; set; }
        public Optimizer optimizer { get; set; }
    }
    public class LayerStorage
    {
        public double[][] tmpActivations  { get; set; }
        public double[][] nodeActivations { get; set; }
        public double[][] nodeDeltas      { get; set; }
    }
    public class Layer
    {
        #region Layer Constructor
        public Layer(Model md, int nodesNum, Activation activation, Optimizer optimizer)
        {
            storage   = new LayerStorage();
            nodes      = new Node[nodesNum];
            totalNodes = nodesNum;
            
            for(int i = 0; i < nodesNum; i++)
            {
                nodes[i] = new Node(activation, optimizer);
            }

            storage.nodeActivations = new double[nodesNum][];
            storage.tmpActivations  = new double[nodesNum][];
            storage.nodeDeltas      = new double[nodesNum][];

            model = md;
        }
        #endregion

        #region Layer Variables
        public Model model { get; set; }
        public int nodeId { get; set; }
        public int totalNodes { get; set; }
        public Node[] nodes { get; set; }
        public LayerStorage storage { get; set; }
        #endregion

        #region Layer Voids
        public void LayerPredict(double[][] inputs)
        {
            for (nodeId = 0; nodeId < totalNodes; nodeId++)
            {
                Node node = nodes[nodeId];
                node.Train(model, inputs);
                storage.tmpActivations[nodeId] = node.storage.activations;
            }
            double[][] tmpActivations = storage.tmpActivations;
            storage.nodeActivations = DataActions.TransposeList(tmpActivations);
        }
        public void LayerDeltas(double[][] prevDerivs, double[][] respectTo)
        {
            for (nodeId = 0; nodeId < totalNodes; nodeId++)
            {
                Node node = nodes[nodeId];
                node.LDeltas(prevDerivs, respectTo);
                storage.nodeDeltas[nodeId] = node.storage.deltas;
            }
        }
        public void UpdateNodes()
        {
            for (nodeId = 0; nodeId < totalNodes; nodeId++)
            {
                Node node = nodes[nodeId];
                node.UpdateGradient(model);
            }
        }
        #endregion
    }
}

//Model.cs
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace AimsMLV2
{
    #region Model
    public class ModelStorage
    {
        public double[] errors { get; set; }
        public double[] errorDerivs { get; set; }
        public double[][] lErrors { get; set; }
        public double[][] lErrorDerivs { get; set; }

        public int errorFeatures { get; set; }
        public void BErrors(double[] data) {
            errors = new double[data.Length];
            errorDerivs = new double[data.Length];
            errorFeatures = data.Length;
        }
        public void BLErrors(double[][] data) {
            lErrors       = new double[data.Length][];
            lErrorDerivs  = new double[data.Length][];
        }
    }
    public class ErrorModel
    {
        public double error { get; set; }
        public double errorDeriv { get; set; }
    }
    public class Model
    {
        #region Model Constructor
        public Model(Errors error = Errors.Mean)
        {
            Epoch = 0;
            Epochs = 0;
            Cost = 0;
            errorType = error;
            keepTrain = true;
            storage = new ModelStorage();
            rand = new Random();
        }
        #endregion

        #region Model Variables
        public Random rand { get; set; }
        public bool keepTrain { get; set; }
        public double Cost { get; set; }
        public Errors errorType { get; set; }
        public ModelStorage storage { get; set; }
        public int Epoch { get; set; }
        public int Epochs { get; set; }
        public int BatchId { get; set; }
        public double Learning { get; set; }
        #endregion

        #region Model Voids
        public void SetLearning(double learning)
        {
            Learning = learning;
        }
        public void SetEpochs(int epochs)
        {
            Epochs = epochs;
        }
        private void _GetCost(bool isLayer = false)
        {
            switch(errorType)
            {
                case Errors.Mean:
                    Cost = (!isLayer ? storage.errors.Sum() : storage.lErrors[0].Sum()) / (2 * storage.errorFeatures);
                    break;
                case Errors.LogLoss:
                    Cost = (!isLayer ? storage.errors.Sum() : storage.lErrors[0].Sum()) / storage.errorFeatures;
                    break;
            }

            if (Cost <= Math.Pow(10, -3))
                keepTrain = false;
        }
        public ErrorModel ErrorVarsCalc(double activation, double target)
        {
            ErrorModel eModel = new ErrorModel();

            switch (errorType)
            {
                case Errors.Mean:
                    eModel.error = Math.Pow(activation - target, 2);
                    eModel.errorDeriv = 2 * (activation - target);
                    break;
                case Errors.LogLoss:
                    eModel.error = target == 1 ? -Math.Log(activation) : -Math.Log(1 - activation);
                    eModel.errorDeriv = target == 1 ? -1 / activation : 1 / (1 - activation);
                    break;
            }
            return eModel;
        }
        public void GetError(double[] activations, double[] targets)
        {
            storage.BErrors(activations);
            for (int i = 0; i < storage.errorFeatures; i++)
            {
                ErrorModel res = ErrorVarsCalc(activations[i], targets[i]);
                storage.errors[i] = res.error;
                storage.errorDerivs[i] = res.errorDeriv;
            }

            _GetCost();
        }
        public void GetLayerError(double[][] layerActivations, double[] targets)
        {
            double[][] activationsT = DataActions.TransposeList(layerActivations);
            storage.BLErrors(activationsT);
            for (int i = 0; i < activationsT.Length; i++)
            {
                storage.BErrors(activationsT[i]);
                for (int j = 0; j < storage.errorFeatures; j++)
                {
                    ErrorModel res = ErrorVarsCalc(activationsT[i][j], targets[j]);
                    storage.errors[j] = res.error;
                    storage.errorDerivs[j] = res.errorDeriv;
                }
                storage.lErrors[i] = storage.errors;
                storage.lErrorDerivs[i] = storage.errorDerivs;
            }
            _GetCost(true);
        }
        #endregion
    }
    #endregion

    #region Dataset
    public class DScaler
    {
        double _GetS(double[] inputArray)
        {
            int size = inputArray.Length;
            double[] s_calculations = new double[size];
            for (int i = 0; i < size; i++)
            {
                double calculation = Math.Pow(inputArray[i] - m, 2) / (size - 1);
                s_calculations[i] = calculation;
            }
            return Math.Sqrt(s_calculations.Sum());
        }

        public DScaler(double[] inputArray)
        {
            min = inputArray.Min();
            max = inputArray.Max();
            m = inputArray.Average();
            s = _GetS(inputArray);
        }

        public double m { get; set; }
        public double s { get; set; }
        public double min { get; set; }
        public double max { get; set; }
    }
    public class Data
    {
        public int id { get; set; }
        public double[] input { get; set; }
        public double target { get; set; }
    }
    public class DatasetVars
    {
        public Scaler scaler { get; set; }
        public DScaler[] scalers { get; set; }
        public Data[] dataset { get; set; }
        public Data[][] batches { get; set; }
        public bool hasBatches { get; set; }
    }
    public class Dataset
    {
        public Dataset(Scaler scaler = Scaler.MinMax)
        {
            vars = new DatasetVars();
            vars.scaler = scaler;
        }
        
        public DatasetVars vars { get; set; }

        #region Dataset Voids

        public void SetData(Data[] dataset, int batchSize)
        {
            Random random = new Random();
            vars.dataset = dataset;
            vars.dataset = vars.dataset.OrderBy(_ => random.Next()).ToArray();
            DataActions.Scale(vars);
            DataActions.GetBatches(batchSize, vars);
        }
        public void BuildDemo(int datasetSize, int batch_size, double multy_var, bool isBinary)
        {

            #region DatasetBuild
            vars.dataset = new Data[datasetSize];
            Random random = new Random();

            for (int i = 0; i < datasetSize; i++)
            {
                int value = i + 1;
                Data data = new Data();
                data.id = value;
                data.input = new double[] { value * 10, -Math.Pow(value, 2) };
                data.target = isBinary ? (datasetSize / multy_var) <= value ? 0 : 1 : value * multy_var;
                vars.dataset[i] = data;
            }
            vars.dataset = vars.dataset.OrderBy(_ => random.Next()).ToArray();
            DataActions.Scale(vars);
            DataActions.GetBatches(batch_size, vars);
            #endregion
        }
        #endregion
    }
    #endregion
}
