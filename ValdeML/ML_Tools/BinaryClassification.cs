using System;
namespace ValdeML
{
	public class BinaryClassification : iML
	{
		public BinaryClassification(double learningRate, int epochs = 0)
		{
			grad = new GradientModel();
			model = new Model(Errors.LogLoss);
			model.Learning = learningRate;
            model.Epochs = epochs;
		}

		public Model model { get; set; }
		public GradientModel grad { get; set; }

        public void _BackPropagate()
        {
            for (int i = 0; i < grad.totalFeatures; i++)
            {
                double jw = grad.weightDeltas[i].Sum() / model.BatchSize;
                double jwp = grad.weightDeltasPow[i].Sum() / model.BatchSize;

                double old_vdw = model.B1 * grad.vdw[i] + (1 - model.B1) * jw;
                double old_sdw = model.B2 * grad.sdw[i] + (1 - model.B2) * jwp;
                grad.vdw[i] = old_vdw;
                grad.sdw[i] = old_sdw;
                double vdw_c = grad.vdw[i] / (1 - Math.Pow(model.B1, model.BatchSize));
                double sdw_c = grad.sdw[i] / (1 - Math.Pow(model.B2, model.BatchSize));

                double tmp_w = grad.w[i] - model.Learning * vdw_c / (Math.Sqrt(sdw_c) + model.e);
                grad.w[i] = tmp_w;
            }

            double j = grad.deltas.Sum() / model.BatchSize;
            double jp = grad.deltasPow.Sum() / model.BatchSize;

            double old_vdb = model.B1 * grad.vdb + (1 - model.B1) * j;
            double old_sdb = model.B2 * grad.sdb + (1 - model.B2) * jp;
            grad.vdb = old_vdb;
            grad.sdb = old_sdb;
            double vdb_c = grad.vdb / (1 - Math.Pow(model.B1, model.BatchSize));
            double sdb_c = grad.sdb / (1 - Math.Pow(model.B2, model.BatchSize));

            double tmp_b = grad.b - model.Learning * vdb_c / (Math.Sqrt(sdb_c) + model.e);
            grad.b = tmp_b;
        }

        public void _CalculateDeltas(double[][] respect_to)
        {
            grad.deltas = new double[model.BatchSize];
            grad.deltasPow = new double[model.BatchSize];
            for (int i = 0; i < model.BatchSize; i++)
            {
                grad.deltas[i] = grad.errorDerivs[i] * grad.activationDerivs[i];
                grad.deltasPow[i] = Math.Pow(grad.errorDerivs[i] * grad.activationDerivs[i], 2);
            }

            double[][] inputsT = Transposer.TransposeList(respect_to);
            grad.weightDeltas = new double[model.BatchSize][];
            grad.weightDeltasPow = new double[model.BatchSize][];
            for (int i = 0; i < grad.totalFeatures; i++)
            {
                double[] weightDArray = new double[grad.totalFeatures];
                double[] weightDArrayPow = new double[grad.totalFeatures];
                for (int j = 0; j < grad.totalFeatures; j++)
                {
                    weightDArray[j] = grad.deltas[j] * inputsT[i][j];
                    weightDArrayPow[j] = Math.Pow(grad.deltas[j] * inputsT[i][j], 2);
                }
                grad.weightDeltas[i] = weightDArray;
                grad.weightDeltasPow[i] = weightDArrayPow;
            }
        }

        public void _Errors(double[] targets)
        {
            grad.errors = new double[model.BatchSize];
            grad.errorDerivs = new double[model.BatchSize];

            for (int i = 0; i < model.BatchSize; i++)
            {
                grad.errors[i] = targets[i] == 1 ? -Math.Log(grad.activations[i]) : -Math.Log(1 - grad.activations[i]);
                grad.errorDerivs[i] = targets[i] == 1 ? -1 / grad.activations[i] : 1 / (1 - grad.activations[i]);
            }

            model.Error = grad.errors.Sum() / model.BatchSize;
            if (model.Error <= Math.Pow(10, -3))
                model.KeepTraining = false;
        }

        public void _Predict(double[][] inputs)
        {
            grad.activations = new double[model.BatchSize];
            grad.activationDerivs = new double[model.BatchSize];

            for (int i = 0; i < model.BatchSize; i++)
            {
                double[] featurePreds = new double[grad.totalFeatures];
                for (int j = 0; j < grad.totalFeatures; j++)
                {
                    featurePreds[j] = grad.w[j] * inputs[i][j];
                }

                double prediction = featurePreds.Sum() + grad.b;
                grad.activations[i] = 1 / (1 + Math.Exp(-prediction));
                grad.activationDerivs[i] = grad.activations[i] * (1 - grad.activations[i]);
            }
        }

        public void _GenerateGradient()
        {
            if(!grad.isReady)
            {
                Random random = new Random();
                grad.w = new double[grad.totalFeatures];
                grad.vdw = new double[grad.totalFeatures];
                grad.sdw = new double[grad.totalFeatures];

                for (int i = 0; i < grad.totalFeatures; i++)
                {
                    grad.w[i] = random.NextDouble() - 0.5;
                    grad.vdw[i] = 0;
                    grad.sdw[i] = 0;
                }

                grad.b = 0;
                grad.vdb = 0;
                grad.sdb = 0;

                grad.isReady = true;
            }
        }

        public void Train(Dataset data)
        {
            do
            {
                model.Epoch++;
                for (model.BatchId = 0; model.BatchId < data.batches.Length; model.BatchId++)
                {
                    if(model.KeepTraining)
                    {
                        Data[] batch = data.batches[model.BatchId];
                        double[][] inputs = batch.Select(x => x.input).ToArray();
                        double[] targets = batch.Select(x => x.target).ToArray();
                        model.BatchSize = batch.Length;
                        grad.totalFeatures = batch[0].input.Length;
                        _GenerateGradient();

                        _Predict(inputs);
                        _Errors(targets);
                        _CalculateDeltas(inputs);
                        _BackPropagate();
                    }
                }
            }
            while (model.Epochs.Equals(0) ? model.Error >= 0 && model.KeepTraining : model.Epoch < model.Epochs);
        }

        public double[] Evaluate(double[] features)
        {
            throw new NotImplementedException();
        }
    }
}

