# pg_stlm_script.jl
echo "Fitting Julia script pg_stlm_script.jl"
/home/jrtipton/julia-1.7.1/bin/julia -t 32 pg_stlm_script.jl > matern_out.txt
retval=$?

if [ $retval -ne 0 ]; then
  echo "Julia script pg_stlm_script.jl failed with $retval" > matern_err.txt
  exit $retval
fi

# pg_stlm_overdispersed.jl
echo "Fitting Julia script pg_stlm_overdispersed_script.jl"
/home/jrtipton/julia-1.7.1/bin/julia -t 32 pg_stlm_overdispersed_script.jl > overdispersed_out.txt
retval=$?

if [ $retval -ne 0 ]; then
  echo "Julia script pg_stlm_overdispersed_script.jl failed with $retval" > overdispersed_err.txt
  exit $retval
fi


# pg_stlm_latent.jl
echo "Fitting Julia script pg_stlm_latent.jl"
/home/jrtipton/julia-1.7.1/bin/julia -t 32 pg_stlm_latent.jl > latent_out.txt
retval=$?

if [ $retval -ne 0 ]; then
  echo "Julia script pg_stlm_latent.jl failed with $retval" > latent_err.txt
  exit $retval
fi
