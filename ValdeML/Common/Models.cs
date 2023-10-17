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
            ErrorType                 = (int)error;
            SelectedError             = error.ToString();
            KeepTraining              = true;
        }
        #endregion

        #region Model Variables
        public int Epoch              = 0;
        public int Epochs             { get; set; }

        public int BatchId            { get; set; }
        public int BatchSize          { get; set; }

        public int ErrorType          { get; set; }
        public string SelectedError   { get; set; }
        public double Error           { get; set; }

        public double[][] Errors      { get; set; }
        public double[][] ErrorDerivs { get; set; }

        public double Learning        { get; set; }
        public bool KeepTraining      { get; set; }

        public double B1              = 0.9;
        public double B2              = 0.999;
        public double e               = Math.Pow(10, -8);

        public string evalText        { get; set; }
        #endregion

        #region Model Voids
        private void _GetError()
        {
            if (ErrorType.Equals(0))
            {
                Error = Errors[0].Sum() / (2 * BatchSize);
            }
            else if (ErrorType.Equals(1))
            {
                Error = Errors[0].Sum() / BatchSize;
            }
        }

        public void CalculateError(double[][] layer_activations, double[]targets)
        {
            double[][] layerActivationsT = Transposer.TransposeList(layer_activations);
            int outerSize = layerActivationsT.Length;
            int innerSize = layerActivationsT[0].Length;

            double[][] tmp_errors_lst = new double[outerSize][];
            double[][] tmp_error_deriv_lst = new double[outerSize][];
            for (int i = 0; i < outerSize; i++)
            {

                double[] tmp_errors = new double[innerSize];
                double[] tmp_derivs = new double[innerSize];
                for (int j = 0; j < innerSize; j++)
                {
                    if (ErrorType.Equals(0))
                    {
                        tmp_errors[j] = Math.Pow(layerActivationsT[i][j] - targets[j], 2);
                        tmp_derivs[j] = 2 * (layerActivationsT[i][j] - targets[j]);
                    }
                    else if (ErrorType.Equals(1))
                    {
                        tmp_errors[j] = targets[j] == 1 ? -Math.Log(layerActivationsT[i][j]) : -Math.Log(1 - layerActivationsT[i][j]);
                        tmp_derivs[j] = targets[j] == 1 ? -1 / layerActivationsT[i][j] : 1 / (1 - layerActivationsT[i][j]);
                    }
                }
                tmp_errors_lst[i]      = tmp_errors;
                tmp_error_deriv_lst[i] = tmp_derivs;
            }
            Errors      = tmp_errors_lst;
            ErrorDerivs = tmp_error_deriv_lst;
            _GetError();
        }
        #endregion
    }

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
        public int? id { get; set; }
        public double[] input { get; set; }
        public double target { get; set; }
    }

    public class Dataset
    {
        public Data[] dataSet { get; set; }
        public Data[][] batches { get; set; }
        public DScaler[] scalers { get; set; }

        #region Dataset Voids
        public void GetBatches(int batchSize, Data[] dataSet)
        {
            int datasetSize = dataSet.Length;
            int totalBatches = datasetSize / batchSize;
            batches = new Data[totalBatches][];

            int bid = 0;
            for (int i = 0; i < datasetSize; i += batchSize)
            {
                if (bid.Equals(totalBatches))
                    continue;
                Data[] batch = dataSet.Skip(i).Take(batchSize).ToArray();
                batches[bid] = batch;
                bid++;
            }
        }

        public void BuildDemo(int datasetSize, int batch_size, double multy_var,
                              Scaler activation, bool isBinary)
        {
            int activationID = (int)activation;

            #region DatasetBuild
            Random random = new Random();
            
            dataSet = new Data[datasetSize];
            

            for (int i = 0; i < datasetSize; i++)
            {
                int value   = i + 1;
                Data data   = new Data();
                data.input  = new double[] { value * 10, -Math.Pow(value, -10) };
                data.target = isBinary ? value <= datasetSize / multy_var ? 0 : 1 : value * multy_var;
                dataSet[i] = data;
            }
            dataSet = dataSet.OrderBy(_ => random.Next()).ToArray();

            GetBatches(batch_size, dataSet);
            #endregion

            #region Dataset Scale
            double[][] inputsT = Transposer.TransposeList(dataSet.Select(x => x.input).ToArray());
            scalers = new DScaler[inputsT.Length];
            double[][] tmp_scaled = new double[inputsT.Length][];

            for (int i = 0; i < inputsT.Length; i++)
            {
                double[] selectedInputT = inputsT[i];
                double[] tmp_scale_calcs = new double[selectedInputT.Length];

                DScaler scaler = new DScaler(selectedInputT);
                scalers[i] = scaler;
                for (int j = 0; j < selectedInputT.Length; j++)
                {
                    double inputT = selectedInputT[j];
                    if (activationID.Equals(0))
                        tmp_scale_calcs[j] = (inputT - scaler.min) / (scaler.max - scaler.min);
                    else if (activationID.Equals(1))
                        tmp_scale_calcs[j] = (inputT - scaler.m) / (scaler.max - scaler.min);
                    else if (activationID.Equals(2))
                        tmp_scale_calcs[j] = (inputT - scaler.m) / scaler.s;
                    else
                        throw new Exception(MLMessages.NA0001);
                }
                tmp_scaled[i] = tmp_scale_calcs;
            }

            double[][] scaledT = Transposer.TransposeList(tmp_scaled);
            for (int i = 0; i < scaledT.Length; i++)
            {
                dataSet[i].input = scaledT[i];
            }
            #endregion

            #endregion
        }
    }
}