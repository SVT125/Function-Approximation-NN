// Function approximation with 1 hidden layer.
// Console arguments are lowerBound, upperBound, numUnits; examples are read in from the Functions holder.
// Removed training, gradient descent, random initialization, and regularization inherited from the original neural network.

import org.apache.commons.math3.linear.*;
import org.apache.commons.math3.analysis.function.Sigmoid;
import java.nio.file.*;
import java.io.*;
import java.util.*;

public class FunctionApproximation {
	static List<ExampleFunctionResult> trainingExamples;
	static List<ExampleFunctionResult> testExamples;
	
	static int numUnits; // the number of rectangles will be numUnits/2
	
	static BlockRealMatrix theta1;
	static BlockRealMatrix theta2;
	static BlockRealMatrix pDerivative1;
	static BlockRealMatrix pDerivative2;
	static BlockRealMatrix act2;
	static BlockRealMatrix biasVector;
	
	static BlockRealMatrix hypothesis = new BlockRealMatrix(1,1); // layer 4, final vector result
	static BlockRealMatrix testHypothesis = new BlockRealMatrix(1,1); // the test hypothesis when calculating the cost
	
	static int numExamples = 0; // number of examples, updated within the methods
	static double lowerBound;
	static double upperBound;
	
	static Function f = new ExponentialFunction();
	public static void main( String[] args ) throws Exception {
		initialize(Double.parseDouble(args[0]),Double.parseDouble(args[1]),Integer.parseInt(args[2]));
		
		System.out.println( "The calculated cost on the training set is: " + calculateCost() );
		
		run();
	}
	
	// Initialize all of the fields.
	public static void initialize(double lower, double upper, int units) {
		trainingExamples = readExamples("C:/Users/James/Programming/Functions/trainingexamples/", "training");
		testExamples = readExamples("C:/Users/James/Programming/Functions/testexamples/", "test");
	
		lowerBound = lower;
		upperBound = upper;
		numUnits = units;
		
		biasVector = new BlockRealMatrix(numUnits,1);
		
		theta1 = new BlockRealMatrix(numUnits,1).scalarAdd(1000d);
		updateBias();
		System.out.println( "Initialized first parameter vector." );
		theta2 = new BlockRealMatrix(1,numUnits);
		manualUpdate( f ); // set the function that implements the Function interface
		System.out.println( "Initialized second parameter vector." );
		System.out.println("--------------------------------------------------------------");
	}
	
	// Runs the finalized hypothesis on all of the test examples.
	public static void run() { 
		for( ExampleFunctionResult efr : testExamples ) {
			hypothesis = forwardPropagation(efr);
			System.out.println("--------------------------------------------------------------");
			System.out.println("The input to the approximation was: " + efr.x.getEntry(0));
			printHypothesis(hypothesis);
			System.out.println("This has an actual error of: " + (hypothesis.getEntry(0,0) - f.getOutput(efr.x.getEntry(0))));
		}
	}
	
	// Manually set the outer weights (h) by taking a Function object and calling it per step.
	public static void manualUpdate(Function f) {
		double multiple = 1;
		for( int i = 0; i < numUnits; i++ ) {
			if( (i+1) % 2 != 0 ) {
				double stepIncrement = (upperBound - lowerBound)/(numUnits/2);
				double argument = stepIncrement * multiple++;
				double result = f.getOutput(argument);
				theta2.setEntry(0,i,result);
			} else
				theta2.setEntry(0,i,-theta2.getEntry(0,i-1));
		}
	}
	
	// Calculates the cost function as the average sum of squared errors.
	public static double calculateCost() {
		double cost = 0;
		for( int i = 0; i < numExamples; i++ ) { 
			testHypothesis = forwardPropagation(trainingExamples.get(i));
			cost += Math.pow(testHypothesis.getEntry(0,0) - trainingExamples.get(i).y.getEntry(0),2);	
		}
		/* regularize theta
		cost += (regularizationRate/(2*numExamples)) * (sumSquaredMatrix(theta1) + 
			sumSquaredMatrix(theta2));
		*/
		cost = ((double)(1)/(double) (numExamples)) * cost;
		return cost;
	}
		
	// Runs forward propagation, given this particular neural network.
	// Can readjust to take arguments of number of units and layers.
	public static BlockRealMatrix forwardPropagation(ExampleFunctionResult ex) {
		BlockRealMatrix z2 = theta1.multiply(convertMatrix(ex.x)).add(biasVector);
		act2 = convertMatrix(sigmoid(new ArrayRealVector(z2.getColumnVector(0))));
		convertStepFunction();
		BlockRealMatrix z3 = theta2.multiply(act2);
		// no bias for the output unit
		return convertMatrix(new ArrayRealVector(z3.getColumnVector(0)));
	}
	
	// Sigmoid function that takes an ArrayRealVector.
	public static ArrayRealVector sigmoid( ArrayRealVector arg ) {
		return arg.mapToSelf( new Sigmoid(1e-323,1-(1e-323)) );
	}
	
	// Updates the outer weights (in this case theta2) so that each pair of activation units have outgoing weights +h and -h respectively.
	// Takes the first unit to be +h and updates the second unit of the pair to be -h.
	public static void updateOuterWeights() {
		double h = 0;
		for( int i = 0; i < theta2.getColumnVector(0).getDimension(); i++ ) {
			if( (i+1) % 2 != 0 )
				h = theta2.getEntry(i,0);
			else
				theta2.setEntry(i,0,-h);
		}
	}
	
	// Initialize the first weight according to the steps; in particular, we adjust the bias which is dependent on the weight such that b = -ws.
	public static void updateBias() {
		double stepIncrement = (upperBound - lowerBound)/(numUnits/2);
		double bias;
		boolean flag = false;
		int index = 1;
		for( int i = 0; i < numUnits; i++ ) {
			if(flag && i != 0) {
				bias = (-theta1.getEntry(i,0)) * (index++) * stepIncrement;
				flag = false;
			}
			else if(!flag && i != 0) {
				bias = (-theta1.getEntry(i,0)) * index * stepIncrement;
				flag = true;
			} else
				bias = 0;
				
			biasVector.setEntry(i,0,bias);
		}
	}
	
	// Simplify the activation unit values to better mimic the step function.
	public static void convertStepFunction() {
		for( int i = 0; i < act2.getColumnVector(0).getDimension(); i++ ) {
			if( act2.getEntry(i,0) > 0.5 )
				act2.setEntry(i,0,1);
			else
				act2.setEntry(i,0,0);
		}
	}
			
	// Read in the examples' input and output by reading their RGB values and the name (per format), stored in a List.
	public static List<ExampleFunctionResult> readExamples(String path, String set ) {
		int loadedExamples = 0;
		List<ExampleFunctionResult> ex = new ArrayList<ExampleFunctionResult>();
		Path dir = Paths.get(path);
		try (DirectoryStream<Path> stream = Files.newDirectoryStream(dir)) {
			for (Path file: stream) {
				BufferedReader br = new BufferedReader(new FileReader(file.toString()));
				String line;
				while ((line = br.readLine()) != null) {
					String[] example = line.split(" ");
					ex.add( new ExampleFunctionResult(Double.parseDouble(example[0]),Double.parseDouble(example[1])) );
					loadedExamples++;
				}
				br.close();	
			}
		} catch (IOException | DirectoryIteratorException x) {
			System.err.println(x);
		}
		if( set.equals( "training" ) )
			numExamples = loadedExamples;
		
		System.out.println( "Loaded in " + loadedExamples + " " + set + " examples." );
		return ex;
	}
	
	// Sums the matrix whose elements have been squared.
	public static double sumSquaredMatrix( BlockRealMatrix mat ) {
		double sum = 0;
		for( int i = 0; i < mat.getRowDimension(); i++ ) {
			for( int j = 0; j < mat.getColumnDimension(); j++ ) {
				sum += Math.pow(mat.getEntry(i,j),2);
			}
		}
		return sum;
	}
	
	// Sums the given vector, returns a double value.
	public static double sumVector( ArrayRealVector vec ) {
		double sum = 0;
		for( int i = 0; i < vec.getDimension(); i++ )
			sum += vec.getEntry(i);
		return sum;
	}
	
	// Print the hypothesis - take the highest value across all the entries and convert to the character.
	// Assumes hypothesis is a 1 x n matrix.
	public static void printHypothesis( BlockRealMatrix hyp ) {
		System.out.println( "The hypothesized output is: " + hyp.getEntry(0,0));
	}
	
	// Auxiliary method to turn an ArrayRealVector to a BlockRealMatrix.
	public static BlockRealMatrix convertMatrix( ArrayRealVector vec ) {
		BlockRealMatrix mat = new BlockRealMatrix(vec.getDimension(),1);
		for( int i = 0; i < vec.getDimension(); i++ )
			mat.setEntry( i, 0, vec.getEntry(i));
		return mat;
	}

	// Auxiliary method to turn a BlockRealMatrix to an ArrayRealVector - works only if one dimension of the matrix = 1.
	public static ArrayRealVector convertVector( BlockRealMatrix mat ) {
		if( mat.getRowVector(0).getDimension() == 1 )
			return new ArrayRealVector(mat.getColumnVector(0));
		else
			return new ArrayRealVector(mat.getRowVector(0));
	}
}