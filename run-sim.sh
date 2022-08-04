# run script with nohup ./run-sim.sh > sim_out.txt &
# pg_stlm_script.jl
echo "Fitting Julia script pg_stlm_script.jl"
# /home/jrtipton/julia-1.7.1/bin/julia -t 32 pg_stlm_script.jl > matern_out.txt
nohup /home/jrtipton/julia-1.7.1/bin/julia -t 32 pg_stlm_script.jl 
retval=$?

if [ $retval -ne 0 ]; then
  # echo "Julia script pg_stlm_script.jl failed with $retval" > matern_err.txt
  echo "Julia script pg_stlm_script.jl failed with $retval" 
  exit $retval
fi

# pg_stlm_overdispersed.jl
echo "Fitting Julia script pg_stlm_overdispersed_script.jl"
# /home/jrtipton/julia-1.7.1/bin/julia -t 32 pg_stlm_overdispersed_script.jl > overdispersed_out.txt
nohup /home/jrtipton/julia-1.7.1/bin/julia -t 32 pg_stlm_overdispersed_script.jl 
retval=$?

if [ $retval -ne 0 ]; then
  # echo "Julia script pg_stlm_overdispersed_script.jl failed with $retval" > overdispersed_err.txt
  echo "Julia script pg_stlm_overdispersed_script.jl failed with $retval" 
  exit $retval
fi


# pg_stlm_latent.jl
echo "Fitting Julia script pg_stlm_latent.jl"
# /home/jrtipton/julia-1.7.1/bin/julia -t 32 pg_stlm_latent_script.jl > latent_out.txt
nohup /home/jrtipton/julia-1.7.1/bin/julia -t 32 pg_stlm_latent_script.jl 
retval=$?

if [ $retval -ne 0 ]; then
  # echo "Julia script pg_stlm_latent_script.jl failed with $retval" > latent_err.txt
  echo "Julia script pg_stlm_latent_script.jl failed with $retval" 
  exit $retval
fi
