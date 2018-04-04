#' @export
#' @method print PUfit
print.PUfit<-function(x,...){
  cat("\nCall: ", deparse(x$call), "\n")
  cat("\nOptimization Method: ", deparse(x$optResult$method), "\n\n")
}
