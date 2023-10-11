using System;
namespace ValdeML
{
	public class BCD
	{
		public void Train(Model model, MMODEL[][] batches)
		{
			while (model.error >= 0)
			{
				for (model.bid = 0; model.bid < batches.Length; model.bid++)
				{
					MMODEL[] batch   = batches[model.bid];
					double[] inputs  = batch.Select(x => x.input[0]).ToArray();
					double[] targets = batch.Select(x => x.target).ToArray();
					model.d = batch.Length;

					model.node1.Predict(inputs);
					model.node2.Predict(model.node1.activations);
					model.node3.Predict(model.node2.activations);

                    model.GetErrors("lls", model.node3.activations, targets);

					model.node3.jders  = new double[model.d];
					model.node3.jwders = new double[model.d];
					for (int i = 0; i < model.d; i++) 
					{
						double error_deriv = model.error_derivs[i] * model.node3.activation_derivatives[i];
						model.node3.jders[i] = error_deriv;
					}
					for (int i = 0; i < model.d; i++)
					{
						double jw = model.node3.jders[i] * model.node2.activations[i];
						model.node3.jwders[i] = jw;
					}

					model.node2.jders  = new double[model.d];
					model.node2.jwders = new double[model.d];
					for (int i = 0; i < model.d; i++)
					{
						double error_deriv = model.node3.jders[i] * model.node2.activation_derivatives[i];
						model.node2.jders[i] = error_deriv;
					}
					for (int i = 0; i < model.d; i++)
					{
						double jw = model.node2.jders[i] * model.node1.activations[i];
						model.node2.jwders[i] = jw;
					}

					model.node1.jders  = new double[model.d];
					model.node1.jwders = new double[model.d];
                    for (int i = 0; i < model.d; i++)
                    {
						double error_der = model.node2.jders[i] * model.node1.activation_derivatives[i];
						model.node2.jders[i] = error_der;
					}
                    for (int i = 0; i < model.d; i++)
                    {
						double jw = model.node1.jders[i] * inputs[i];
						model.node1.jwders[i] = jw;
					}

                    //model.node3.GetDerivs(model.error_derivs, model.node2.activations);
                    //model.node2.GetDerivs(model.node3.jders, model.node1.activations);
                    //model.node1.GetDerivs(model.node2.jders, inputs);

                    model.node3.UpdateGradient(model);
                    model.node2.UpdateGradient(model);
                    model.node1.UpdateGradient(model);

					if (model.error <= Math.Pow(10, -2))
						break;
				}

				if (model.error <= Math.Pow(10, -2))
					break;

				var xxx = 10;
				model.old_error = model.error;
			}
		}
	}
}

