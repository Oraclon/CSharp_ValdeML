﻿using System;
namespace ValdeML
{
    public class LoopRetModel
    {
        public double[][] inputs { get; set; }
        public double[] targets { get; set; }
    }
    public class GradientModel
    {
        public GradientModel()
        {
            isReady = false;
            keepTraining = true;
        }

        public bool isReady { get; set; }
        public bool keepTraining { get; set; }
        public LoopRetModel loopRet { get; set; }

        public double[] w { get; set; }
        public double[] vdw { get; set; }
        public double[] sdw { get; set; }

        public double b { get; set; }
        public double vdb { get; set; }
        public double sdb { get; set; }

        public int itemId { get; set; }
        public int batchId { get; set; }
        public int dataSize { get; set; }
        public int featuresLen { get; set; }

        public double[] activations { get; set; }
        public double[] activationDerivs { get; set; }

        public double[] errors { get; set; }
        public double[] errorDerivs { get; set; }

        public double[] deltas { get; set; }
        public double[] deltasPow { get; set; }
        public double[][] wDeltas { get; set; }
        public double[][] wDeltasPow { get; set; }

        private void _BuildGradient()
        {
            Random rand = new Random();
            w = new double[featuresLen];
            vdw = new double[featuresLen];
            sdw = new double[featuresLen];

            for (int i = 0; i < featuresLen; i++)
            {
                w[i] = rand.NextDouble() - .5;
                vdw[i] = 0;
                sdw[i] = 0;
            }

            b = 0;
            vdb = 0;
            sdb = 0;

            isReady = true;
        }
        public LoopRetModel GetLoopData(Dataset dataSet)
        {
            Data[] data = dataSet.hasBatches ? dataSet.batches[batchId] : dataSet.dataSet;
            dataSize = data.Length;
            featuresLen = data[0].input.Length;

            loopRet = new LoopRetModel();
            loopRet.inputs = data.Select(x => x.input).ToArray();
            loopRet.targets = data.Select(x => x.target).ToArray();

            if (!isReady)
                _BuildGradient();

            return loopRet;
        }
    }

    public class Model
    {
        #region Model Constructor
        public Model(Errors error)
        {
            ErrorType                 = (int)error;
            SelectedError             = error.ToString();
            KeepTraining              = true;
            Error = 0;
        }
        #endregion

        #region Model Variables
        public int Epoch              = 0;
        public int Epochs             { get; set; }

        public int ItemId             { get; set; }
        public int BatchId            { get; set; }
        public int BatchSize          { get; set; }
        public int FeatureSize        { get; set; }

        public int ErrorType          { get; set; }
        public string SelectedError   { get; set; }
        public double Error           { get; set; }

        public double[][] Errors      { get; set; }
        public double[][] ErrorDerivs { get; set; }

        public double Learning        { get; set; }
        public bool KeepTraining      { get; set; }
        public bool IsReady           { get; set; }

        public double B1              = 0.9;
        public double B2              = 0.999;
        public double e               = Math.Pow(10, -8);
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

            if (Error <= Math.Pow(10, -3))
                KeepTraining = false;
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
        public bool hasBatches { get; set; }

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

            hasBatches = true;
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