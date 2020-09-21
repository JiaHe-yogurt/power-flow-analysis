##LR adaboost
adaboost_LR = function(X, y, n_rounds = 30,stopping=0.001 , verbose = FALSE){
  
  # check data types
  if(!all(y %in% c(-1,1)))
    stop("y must take values in -1, 1")
  n = dim(X)[1]
  w = rep(1/n, n)
  N_pos=table(y)[2]
  N_neg=table(y)[1] 
  trees = list()
  alphas = list()
  
  for(i in seq(n_rounds)){
    #classifier
    
    fit<-glm(y ~ ., data =X,family=binomial,weights=w)
    probs=predict(fit,data.frame(X),type="response")
    pred=rep("-1",dim(X)[1])
    pred[probs>.5]="1"
    
    e = sum(w*(pred != y))
    
    error =pred != as.numeric(as.character(y))
    e_pos=sum(error[which(y=="1")])/N_pos
    e_neg=sum(error[which(y=="-1")])/N_neg
    
    if (abs(e) < 1e-08) {
      if (i == 1) {
        trees[[i]] = fit
        alphas[[i]] = 1
        terms = fit$terms
        break
      }
      break
    }
    
    alpha = 1/2*log((1-e)/e)
    w = w*exp(-alpha*as.integer(as.character(pred))*as.integer(as.character(y)))
    w = w/sum(w)
    
    if (i == 1) {
      terms = fit$terms
    }
    else {
      fit$terms = NULL
    }
    trees[[i]] = fit
    alphas[[i]] = alpha
    if (verbose & (i%%10 == 0)) 
      cat("Iteration: ", i, "\n")
    
  }
  
  #normalize class weight for WLR use
  out = list(alphas= unlist(alphas),trees = trees, terms = terms)
  class(out) = "adaboost"
  Predict = Predict.adaboost_LR(out, X,y,type="response")
  out$confusion_matrix=table(Predict,y)
  out
}


Predict.adaboost_LR=
  function (object, X, y,type = c("response", "prob"), n_tree = NULL, 
            ...) 
  {
    type = match.arg(type)
    if (is.null(n_tree)) {
      tree_seq = seq_along(object$alphas)
    }
    else {
      if (n_tree > length(object$alphas)) 
        stop("n_tree must be less than the number of trees used in fit")
      tree_seq = seq(1, n_tree)
    }
    f=0
    
    for (i in 1:length(object$trees)) {
      tree = object$trees[[i]]
      tree$terms = object$terms
      Pred= predict(tree, X,type = "response")
      pred=rep("-1",dim(X)[1])
      pred[Pred>.5]="1"
      f = f + object$alphas[i] * as.integer(pred)
          
    }
    if (type == "response") {
      sign(f)
    }
    else if (type == "prob") {
      1/(1 + exp(-2 * f))
    }
  }



model<-adaboost_LR(X1,y,n_rounds = T)
train_FN[T]=model$confusion_matrix[1,2]/sum(model$confusion_matrix[,2])
CW[T,]=model$CW

