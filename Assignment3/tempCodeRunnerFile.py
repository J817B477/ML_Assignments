 print(f"R^2: {ridge_poly_model['R_squared']}")
  print(f"MSE: {ridge_poly_model['mse']}")
  print(f"\nModel Coefficients: \n{ridge_poly_model['model'].coef_}")
  print(f"\nIntercept: {ridge_poly_model['model'].intercept_}")

  # Scatter plot y_test vs y_pred
  plt.scatter(x=y_test, y=ridge_poly_model['predictions'], alpha=0.6)
  plt.plot(y_test, y_test, color='black', linestyle='--')
  plt.suptitle("Actual vs Predicted: Ridge Regression", fontsize=14)
  plt.title("(degree 2 polynomial features)", fontsize=12)
  plt.xlabel("Actual Home Price", fontsize=11)
  plt.ylabel("Predicted Price", fontsize=11)
  plt.grid()
  plt.savefig("deg2polyRidge_actualVpredicted.png", dpi=300)
  plt.show()
  plt.close()

  # Scatter plot y_test vs residuals
  plt.scatter(x=y_test, y=ridge_poly_model['residuals'], alpha=0.6)
  plt.axhline(y=np.mean(ridge_poly_model['residuals']), color='black', linestyle='--', linewidth=1)
  plt.suptitle("Ridge Regression Residuals", fontsize=14)
  plt.title("(degree 2 polynomial features)", fontsize=12)
  plt.xlabel("True Prices")
  plt.ylabel("Price Estimation Errors")
  plt.grid()
  plt.savefig("deg2polyRidge_residualsPlot.png")
  plt.show()
  plt.close()

  # Individual feature curves
  for name in col_names:
      if X_train[name].dtype in ["object", "category"]:
          continue

      simple_model = make_linear_model(
          pd.DataFrame(X_train[name]), 
          pd.DataFrame(X_test[name]),
          y_train,
          y_test,
          degree=2,
          ridge=True,
          alpha=1.0
      )

      coef = simple_model['model'].coef_
      intercept = simple_model['model'].intercept_

      x_values = X_train[name]
      x_sorted = np.sort(x_values.values).reshape(-1, 1)

      poly = PolynomialFeatures(degree=2, include_bias=False)
      x_poly = poly.fit_transform(x_sorted)

      y_values = x_poly @ coef + intercept

      plt.scatter(x_values, y_train, alpha=0.6, label="Train Data")
      plt.plot(x_sorted, y_values, color="red", label="Fitted Polynomial Curve")
      plt.suptitle(f"Fitted Curve against {name}/House_Price", fontsize=14)
      plt.title(f"(polynomial degree = 2, ridge regression)", fontsize=12)
      plt.xlabel(name)
      plt.ylabel("House_Price")
      plt.legend()
      plt.savefig(f"{name}_simpleRidgeRegression_2rigdepoly.png")
      plt.show()
      plt.close()