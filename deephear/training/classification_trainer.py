import torch

def train_classification(
    model, train_loader, val_loader, criterion, optimizer,
    num_epochs, device, save_path=None
):
    model.to(device)
    best_acc = 0.0
    for ep in range(1, num_epochs+1):
        # train
        model.train()
        running_loss = 0; correct = 0; total = 0
        for X,y in train_loader:
            X,y = X.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(X)
            loss   = criterion(logits,y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()*X.size(0)
            preds = logits.argmax(1)
            correct   += (preds==y).sum().item()
            total     += X.size(0)
        train_loss = running_loss/total
        train_acc  = correct/total

        # validate
        model.eval()
        val_loss=0; val_corr=0; val_tot=0
        with torch.no_grad():
            for X,y in val_loader:
                X,y = X.to(device), y.to(device)
                logits= model(X)
                loss  = criterion(logits,y)
                val_loss+= loss.item()*X.size(0)
                preds   = logits.argmax(1)
                val_corr+= (preds==y).sum().item()
                val_tot += X.size(0)
        val_loss /= val_tot
        val_acc   = val_corr/val_tot

        print(f"Ep{ep}: Train {train_loss:.4f}/{train_acc:.4f}  Val {val_loss:.4f}/{val_acc:.4f}")
        if save_path and val_acc>best_acc:
            torch.save(model.state_dict(), save_path)
            best_acc=val_acc
    return model