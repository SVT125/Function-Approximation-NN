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

// An interface for a univariate function accepting a single input and output.
interface Function {
	public double getOutput( double input );
}

class IdentityFunction implements Function { 
	public double getOutput( double input ) { return input; }
}

class ScalarSineFunction implements Function {
	public double getOutput( double input ) {
		return 20 * Math.sin(input);
	}
}

class ExponentialFunction implements Function {
	public double getOutput( double input ) {
		return Math.pow(Math.E,input);
	}
}

