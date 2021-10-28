function  [ delta ] =   Sample_delta(X_train,Theta,epilson0,Station)
    delta = ones(size(Theta,2),1);
    if Station==0
        shape = epilson0+sum(X_train,1);
        scale = epilson0+sum(Theta,1);
        delta = gamrnd(shape,1./scale)';
    else
        shape = epilson0+sum(sum(X_train));
        scale = epilson0+sum(sum(Theta));
       delta(:) = gamrnd(shape,1./scale)';
    end
end
