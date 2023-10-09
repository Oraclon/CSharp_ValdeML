using System;
namespace ValdeML
{
	public class Momentum
	{
		public double Optimize(Grad grad, bool isbias)
		{
			double optimized_var= 0.0;

			if(!isbias)
			{
				Wopt wop = grad.ws[grad.fid];
				double old_vdw = grad.b1 * wop.vdw + (1 - grad.b1) * grad.GetJW();
				wop.vdw = old_vdw;
				optimized_var = wop.w - grad.a * wop.vdw;
			}
			else
			{
				Bopt bop = grad.b;
				double old_vdb = grad.b1 * bop.vdb + (1 - grad.b1) * grad.GetJB();
				bop.vdb = old_vdb;
				optimized_var = bop.b - grad.a * bop.vdb;
			}

			return optimized_var;
		}
	}

	public class RMSProp
	{
		public double Optimize(Grad grad, bool isbias)
		{
            double optimized_var = 0.0;

			if(!isbias)
			{
				Wopt wop = grad.ws[grad.fid];
				double old_sdw = grad.b2 * wop.sdw + (1 - grad.b2) * Math.Pow(grad.GetJW(), 2);
				wop.sdw = old_sdw;
				double sdw_c = wop.sdw / (1 - Math.Pow(grad.b2, grad.d));
				optimized_var = wop.w - grad.a * grad.GetJW() / (Math.Sqrt(sdw_c) + grad.e);
			}
			else
			{
				Bopt bop = grad.b;
				double old_sdb = grad.b2 * bop.sdb + (1 - grad.b2) * Math.Pow(grad.GetJB(), 2);
				bop.sdb = old_sdb;
				double sdb_c = bop.sdb / (1 - Math.Pow(grad.b2, grad.d));
				optimized_var = bop.b - grad.a * grad.GetJB() / (Math.Sqrt(sdb_c) + grad.e);
			}

			return optimized_var;
        }
	}

	public class Adam
	{
		public double Optimize(Grad grad, bool isbias)
		{
			double optimized_var = 0.0;

			if(!isbias)
			{
				Wopt wop = grad.ws[grad.fid];

				double old_vdw = grad.b1 * wop.vdw + (1 - grad.b1) * grad.GetJW();
				wop.vdw = old_vdw;
				double vdw_c = wop.vdw / (1 - Math.Pow(grad.b1, grad.d));

				double sqrd_deriv = grad.input_derivs.Select(x => Math.Pow(x, 2)).ToArray().Sum() / grad.d;
				//double sqrd_deriv = grad.input_derivs.Select(x => Math.Pow(x, 2)).ToArray().Sum();
				//double sqrd_deriv = Math.Pow(grad.GetJW(), 2);

				double old_sdw = grad.b2 * wop.sdw + (1 - grad.b2) * sqrd_deriv;
				wop.sdw = old_sdw;
				double sdw_c = wop.sdw / (1 - Math.Pow(grad.b2, grad.d));

				optimized_var = wop.w - grad.a * vdw_c / (Math.Sqrt(sdw_c) + grad.e);
			}
			else
			{
				Bopt bop = grad.b;

				double old_vdb = grad.b1 * bop.vdb + (1 - grad.b1) * grad.GetJB();
				bop.vdb = old_vdb;
				double vdb_c = bop.vdb / (1 - Math.Pow(grad.b1, grad.d));

				double sqrd_deriv = grad.derivs.Select(x => Math.Pow(x, 2)).ToArray().Sum() / grad.d;
				//double sqrd_deriv = grad.derivs.Select(x => Math.Pow(x, 2)).ToArray().Sum();
				//double sqrd_deriv = Math.Pow(grad.GetJB(), 2);

				double old_sdb = grad.b2 * bop.sdb + (1 - grad.b2) * sqrd_deriv;

				//double old_sdb = grad.b2 * bop.sdb + (1 - grad.b2) * Math.Pow(grad.GetJB(), 2);

				bop.sdb = old_sdb;
				double sdb_c = bop.sdb / (1 - Math.Pow(grad.b2, grad.d));

				optimized_var = bop.b - grad.a * vdb_c / (Math.Sqrt(sdb_c) + grad.e);
			}

			return optimized_var;
		}
	}
}

