// Trivial fragment shader for the depth-only shadow pass. Writes no color (the pipeline has no
// color attachment); depth is written automatically. Some drivers/pipelines still require a
// fragment stage, so provide an empty one.
void main() {}
