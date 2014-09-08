import org.apache.commons.math3.linear.*;

class ExampleFunctionResult {
	ArrayRealVector y = new ArrayRealVector(1);
	ArrayRealVector x = new ArrayRealVector(1);

	
	// Construct an example 
	public ExampleFunctionResult( double input, double output ) {
		x.setEntry(0,input);
		y.setEntry(0,output);
	}
}