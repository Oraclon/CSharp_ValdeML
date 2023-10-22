using System;
namespace ValdeML
{
	public class MLCore
	{
		public Model model { get; set; }
		public GradientModel grad { get; set; }


		public MLCore()
		{
			model = new Model(Errors.None);
			grad = new GradientModel();
		}

		#region Abstract Voids
		public virtual void ActivatePrediction(double prediction)
		{
			grad.activations[grad.itemId] = prediction;
			grad.activationDerivs[grad.itemId] = 1;
		}
		public virtual void CalculateErrors(double activation, double target)
		{
			grad.errors[grad.itemId] = Math.Pow(activation - target, 2);
			grad.errorDerivs[grad.itemId] = 2 * (activation - target);
		}
        #endregion

		private void _UpdateGradient()
		{
			for (int i = 0; i < grad.featuresLen; i++)
			{
				double tmp_w = 0;
				double jw = grad.wDeltas[i].Sum() / grad.dataSize;
				double jwp = grad.wDeltasPow[i].Sum() / grad.dataSize;

				double old_vdw = model.B1 * grad.vdw[i] + (1 - model.B1) * jw;
				double old_sdw = model.B2 * grad.sdw[i] + (1 - model.B2) * jwp;
				grad.vdw[i] = old_vdw;
				grad.sdw[i] = old_sdw;
				double vdw_c = grad.vdw[i] / (1 - Math.Pow(model.B1, grad.dataSize));
				double sdw_c = grad.sdw[i] / (1 - Math.Pow(model.B2, grad.dataSize));

				if (model.ErrorType.Equals(0)) 
					tmp_w = grad.w[i] - model.Learning * jw;
				else
					tmp_w = grad.w[i] - model.Learning * vdw_c / (Math.Sqrt(sdw_c) + model.e);
				grad.w[i] = tmp_w;
			}

			double tmp_b = 0;
			double j = grad.deltas.Sum() / grad.dataSize;
			double jp = grad.deltasPow.Sum() / grad.dataSize;

			double old_vdb = model.B1 * grad.vdb + (1 - model.B1) * j;
			double old_sdb = model.B2 * grad.sdb + (1 - model.B2) * jp;
			grad.vdb = old_vdb;
			grad.sdb = old_sdb;
			double vdb_c = grad.vdb / (1 - Math.Pow(model.B1, grad.dataSize));
			double sdb_c = grad.sdb / (1 - Math.Pow(model.B2, grad.dataSize));

			if(model.ErrorType.Equals(0))
				tmp_b = grad.b - model.Learning * j;
			else
				tmp_b = grad.b - model.Learning * vdb_c / (Math.Sqrt(sdb_c) + model.e);
			grad.b = tmp_b;
		}

        private void _Deltas(double[][] respectTo)
        {
			grad.deltas     = new double[grad.dataSize];
			grad.deltasPow  = new double[grad.dataSize];
            grad.wDeltas    = new double[grad.featuresLen][];
			grad.wDeltasPow = new double[grad.featuresLen][];

            for (int i = 0; i < grad.dataSize; i++)
			{
				grad.deltas[i]    = grad.errorDerivs[i] * grad.activationDerivs[i];
				grad.deltasPow[i] = Math.Pow(grad.errorDerivs[i] * grad.activationDerivs[i], 2);
			}

			double[][] inputsT = Transposer.TransposeList(respectTo);

			for (int i = 0; i < grad.featuresLen; i++)
			{
				double[] wFeatureCalcs    = new double[grad.dataSize];
				double[] wFeatureCalcsPow = new double[grad.dataSize];
				for (int j = 0; j < grad.dataSize; j++)
				{
					wFeatureCalcs[j]    = grad.deltas[j] * inputsT[i][j];
					wFeatureCalcsPow[j] = Math.Pow(grad.deltas[j] * inputsT[i][j], 2);
				}
				grad.wDeltas[i]    = wFeatureCalcs;
				grad.wDeltasPow[i] = wFeatureCalcsPow;
			}
		}

        private void _Errors(double[] targets)
		{
			grad.errors = new double[grad.dataSize];
			grad.errorDerivs = new double[grad.dataSize];

			for (grad.itemId = 0; grad.itemId < grad.dataSize; grad.itemId++)
			{
				CalculateErrors(grad.activations[grad.itemId], targets[grad.itemId]);
			}

			if (model.ErrorType.Equals(0))
				model.Error = grad.errors.Sum() / (2 * grad.dataSize);
			else
				model.Error = grad.errors.Sum() / grad.dataSize;

			if (model.Error <= Math.Pow(10, -3))
				grad.keepTraining = false;
		}

		private void _Predict(double[][] inputs)
		{
			grad.activations = new double[grad.dataSize];
			grad.activationDerivs = new double[grad.dataSize];

			for (grad.itemId = 0; grad.itemId < grad.dataSize; grad.itemId++)
			{
				double[] featuresCalcs = new double[grad.featuresLen];
				for (int i = 0; i < grad.featuresLen; i++)
				{
					featuresCalcs[i] = grad.w[i] * inputs[grad.itemId][i];
				}

				double prediction = featuresCalcs.Sum() + grad.b;
				ActivatePrediction(prediction);
			}
		}

		public void Train(Dataset dataSet)
		{
			do
			{
				model.Epoch++;
				for (grad.batchId = 0; grad.batchId < dataSet.batches.Length; grad.batchId++)
				{
					if (grad.keepTraining)
					{
						LoopRetModel loopRet = grad.GetLoopData(dataSet);
						_Predict(loopRet.inputs);
						_Errors(loopRet.targets);
						_Deltas(loopRet.inputs);
						_UpdateGradient();
					}
					else
						break;
				}
			}
			while (model.Error >= 0 && grad.keepTraining);
		}
	}

	public class LinearRegression: MLCore
	{
		public LinearRegression()
		{
			model = new Model(Errors.Mean);
		}
	}

	public class BinaryClassification: MLCore
	{
		public BinaryClassification()
		{
			model = new Model(Errors.LogLoss);
		}

        public override void ActivatePrediction(double prediction)
        {
			double activation = 1 / (1 + Math.Exp(-prediction));
			grad.activations[grad.itemId] = activation;
			grad.activationDerivs[grad.itemId] = activation * (1 - activation);
        }

        public override void CalculateErrors(double activation, double target)
        {
			grad.errors[grad.itemId] = target == 1 ? -Math.Log(activation) : -Math.Log(1 - activation);
			grad.errorDerivs[grad.itemId] = target == 1 ? -1 / activation : 1 / (1 - activation);
        }
    }
}

