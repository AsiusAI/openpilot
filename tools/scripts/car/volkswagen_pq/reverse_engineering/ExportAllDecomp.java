// Export every discovered function as decompiled C for offline searching.
//@category Analysis

import java.io.File;
import java.io.PrintWriter;

import ghidra.app.decompiler.DecompInterface;
import ghidra.app.decompiler.DecompileResults;
import ghidra.app.script.GhidraScript;
import ghidra.program.model.listing.Function;
import ghidra.program.model.listing.FunctionIterator;

public class ExportAllDecomp extends GhidraScript {
  @Override
  protected void run() throws Exception {
    String[] args = getScriptArgs();
    if (args.length != 1) {
      throw new IllegalArgumentException("usage: ExportAllDecomp <output-file>");
    }

    DecompInterface decompiler = new DecompInterface();
    decompiler.toggleCCode(true);
    decompiler.toggleSyntaxTree(true);
    if (!decompiler.openProgram(currentProgram)) {
      throw new IllegalStateException("could not open program in decompiler");
    }

    try (PrintWriter out = new PrintWriter(new File(args[0]))) {
      FunctionIterator functions = currentProgram.getFunctionManager().getFunctions(true);
      int count = 0;
      while (functions.hasNext() && !monitor.isCancelled()) {
        Function function = functions.next();
        out.printf("\n/* ===== %s @ %s ===== */\n", function.getName(), function.getEntryPoint());
        DecompileResults result = decompiler.decompileFunction(function, 60, monitor);
        if (result.decompileCompleted()) {
          out.println(result.getDecompiledFunction().getC());
        } else {
          out.printf("/* decompile failed: %s */\n", result.getErrorMessage());
        }
        count++;
      }
      println("Exported " + count + " functions to " + args[0]);
    } finally {
      decompiler.dispose();
    }
  }
}
