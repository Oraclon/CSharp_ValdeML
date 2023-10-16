namespace ValdeML
{
    class Program
    {
        enum Fruits
        {
            orange,
            grapefruit
        }
        static void Main(string[] args)
        {
            DatasetMultFeatures data = new DatasetMultFeatures();
            data.Build(2000000, 512, 2, "zscore", true);
            MMODEL[][] toTrain = data.batches.Take(data.batches.Length - 10).ToArray();
            MMODEL[][] toEval = data.batches.Skip(data.batches.Length - 10).ToArray();


            Layer layer1 = new Layer(18, Activation.Tanh);
            Layer layer2 = new Layer(18, Activation.Tanh);
            Layer layer3 = new Layer(1, Activation.Sigmoid);

            Model model = new Model(Errors.LogLoss);
            model.Learning = .4;

            while (model.Error >= 0)
            {
                model.Epoch++;
                for (model.BatchId = 0; model.BatchId < toTrain.Length; model.BatchId++)
                {
                    MMODEL[] batch = toTrain[model.BatchId];
                    double[][] inputs = batch.Select(x => x.input).ToArray();
                    double[] targets = batch.Select(x => x.target).ToArray();
                    model.BatchSize = batch.Length;

                    layer1.NodesPredict(inputs);
                    layer2.NodesPredict(layer1.nodeActivations);
                    layer3.NodesPredict(layer2.nodeActivations);

                    model.CalculateError(layer3.nodeActivations, targets);

                    layer3.NodesCalcDeltas(model.ErrorDerivs, layer2.nodeActivations);
                    layer2.NodesCalcDeltas(layer3.nodeDeltas, layer1.nodeActivations);
                    layer1.NodesCalcDeltas(layer2.nodeDeltas, inputs);

                    layer3.NodesUpdate(model);
                    layer2.NodesUpdate(model);
                    layer1.NodesUpdate(model);

                    if (model.Error <= Math.Pow(10, -3))
                        break;
                }

                if (model.Error <= Math.Pow(10, -3))
                {
                    EvalTraining(toEval);
                    break;
                }

                string res = $"{model.Epoch}, {model.BatchId}, {model.Error}";
                Console.Write("\r{0} ", res);
            }

            void EvalTraining(MMODEL[][] toEval)
            {
                int correct = 0;
                int wrong = 0;
                double acc = 0;
                //I know i shouldn't use this. But i needed a fast solution for that.
                foreach (MMODEL[] batch in toEval)
                {
                    foreach(MMODEL item in batch)
                    {
                        layer1.NodesEvaluate(item.input);
                        layer2.NodesEvaluate(layer1.evalActivations);
                        layer3.NodesEvaluate(layer2.evalActivations);

                        var prediction = (double)Math.Round(layer3.evalActivations[0]);
                        if (item.target.Equals(prediction))
                            correct++;
                        else
                            wrong++;
                    }
                }
                model.evalText = $"{correct} {wrong} {(correct * 100) / (correct+ wrong)}%";
            }
        }
    }
}
