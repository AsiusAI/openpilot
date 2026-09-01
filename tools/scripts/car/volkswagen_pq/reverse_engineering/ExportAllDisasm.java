// Export disassembly for every discovered function.
//@category Analysis

import java.io.File;
import java.io.PrintWriter;

import ghidra.app.script.GhidraScript;
import ghidra.program.model.listing.Function;
import ghidra.program.model.listing.FunctionIterator;
import ghidra.program.model.listing.Instruction;
import ghidra.program.model.listing.InstructionIterator;

public class ExportAllDisasm extends GhidraScript {
  @Override
  protected void run() throws Exception {
    String[] args = getScriptArgs();
    if (args.length != 1) {
      throw new IllegalArgumentException("usage: ExportAllDisasm <output-file>");
    }
    try (PrintWriter out = new PrintWriter(new File(args[0]))) {
      FunctionIterator functions = currentProgram.getFunctionManager().getFunctions(true);
      int count = 0;
      while (functions.hasNext() && !monitor.isCancelled()) {
        Function function = functions.next();
        out.printf("\n; ===== %s @ %s =====\n", function.getName(), function.getEntryPoint());
        InstructionIterator instructions = currentProgram.getListing().getInstructions(function.getBody(), true);
        while (instructions.hasNext()) {
          Instruction instruction = instructions.next();
          out.printf("%s  %-10s %s\n", instruction.getAddress(), instruction.getMnemonicString(), instruction.toString().substring(instruction.getMnemonicString().length()).trim());
        }
        count++;
      }
      println("Exported " + count + " functions to " + args[0]);
    }
  }
}
