# Takes a csv with var, val and color columns and returns a list with a vector per var
make_color_vectors <- function(colors_df) {
  sapply(unique(colors_df$var), make_color_vector, colors_df)
}

# Creates a named vector for a specific var in the var_colors.csv
make_color_vector <- function(var, colors_df) {
  colors <- colors_df %>% filter(var == !!var) %>% pull(color)
  names(colors) <- colors_df %>% filter(var == !!var) %>% pull(val)
  return(colors)
}
