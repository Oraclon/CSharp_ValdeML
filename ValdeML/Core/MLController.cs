using System;
namespace ValdeML
{
	public class LayerModel
	{
		public LayerModel(Activation node_activation, int total_nodes)
		{
			activation = node_activation;
			totalNodes = total_nodes;
		}
		internal Activation activation { get; set; }
		internal int totalNodes { get; set; }
	}

	public class MLController
	{
		public int totalLayers { get; set; }

		public List<LayerModel> LayersOptions = new List<LayerModel>();
		public Model? model { get; set; }
		public Layer[] Layers { get; set; }

		public void AddLayer(Activation activation, int total_nodes)
		{
			LayerModel layer = new LayerModel(activation, total_nodes);
            LayersOptions.Add(layer);
		}

		public void BuildLayers()
		{
			totalLayers = LayersOptions.Count();
			Layers = new Layer[totalLayers];

			for (int i = 0; i < totalLayers; i++)
			{
				LayerModel lm = LayersOptions[i];
				Layer layer = new Layer(i + i, lm.totalNodes, lm.activation);
				Layers[i] = layer;
			}
		}

		public void StartTraining(Model model, Dataset dataSet)
		{

			if (dataSet.hasBatches)
			{
				do
				{
					model.Epoch++;

					for (model.BatchId = 0; model.BatchId < dataSet.batches.Length; model.BatchId++)
					{
						if (model.KeepTraining)
						{
							Data[] batch = dataSet.batches[model.BatchId];
							double[][] inputs = batch.Select(x => x.input).ToArray();
							double[] targets = batch.Select(x => x.target).ToArray();
							model.BatchSize = batch.Length;
							int layersSize = totalLayers - 1;

							for (int i = 0; i < totalLayers; i++)
							{
								if (i.Equals(0))
									Layers[i].NodesPredict(inputs);
								else
									Layers[i].NodesPredict(Layers[i - 1].nodeActivations);
							}

							model.CalculateError(Layers.Last().nodeActivations, targets);

							for (int i = layersSize; i >= 0; i--)
							{
								if (i.Equals(layersSize))
									Layers[i].NodesCalcDeltas(model.ErrorDerivs, Layers[i - 1].nodeActivations);
								else if (i.Equals(0))
									Layers[i].NodesCalcDeltas(Layers[i + 1].nodeDeltas, inputs);
								else
									Layers[i].NodesCalcDeltas(Layers[i + 1].nodeDeltas, Layers[i - 1].nodeActivations);
							}

							for (int i = layersSize; i >= 0; i--)
							{
								Layers[i].NodesUpdate(model);
							}
						}
						else
							break;
					}
				}
				while (model.Epochs.Equals(0) ? model.Error >= 0 && model.KeepTraining : model.Epoch < model.Epochs);
			}

		}
	}
}

